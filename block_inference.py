#!/usr/bin/env python3
"""
block_inference.py -- block-aware re-analysis of the paired L0 vs L1 comparison.

Why this exists
---------------
Proposition 2 of the paper proves that under deterministic enumerated decoding
failures arrive in BLOCKS: tasks sharing a decision context all receive the same
instruction and therefore all fail together. Exact McNemar and the task-level
sign-flip permutation test both condition on the discordant pairs and treat them
as EXCHANGEABLE. That assumption is exactly what Proposition 2 denies, and it is
violated in the direction that inflates significance.

This script recomputes the primary endpoint without that assumption. It needs no
GPU and no model: it runs off the per-task records already released with the
paper.

Input
-----
A CSV (or JSON records) with one row per (task, language) with at least:

    task_id      any hashable id, paired across languages
    language     e.g. "L0", "L1", "L0prime"
    success      0/1

and, for the block-aware tests, ONE of:

    block        explicit decision-context key (best: the argmax-invariant
                 prefix, e.g. f"{legal_action_set_signature}|{step1_argmax}")
    requires_message   0/1  (the paper's own 32/8 split; usable proxy)

If neither is present the script falls back to one-task-per-block and says so
loudly -- that fallback reproduces the numbers already in the paper and is not
the point of running this.

Usage
-----
    python block_inference.py records.csv --arm-a L0 --arm-b L1
    python block_inference.py records.csv --arm-a L0prime --arm-b L1
    python block_inference.py records.csv --block-col requires_message

What to put in the paper
------------------------
Report `p_block_permutation` and `n_effective` alongside the exact McNemar
value, and say that the exact value assumes exchangeability the design violates.
Proposition 2 predicts exactly this correlation structure, so it must be tested rather than assumed away.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_records(path):
    if path.endswith(".json") or path.endswith(".jsonl"):
        with open(path) as fh:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in fh if line.strip()]
            data = json.load(fh)
        return data["records"] if isinstance(data, dict) else data

    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def to_int(v):
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n", ""):
        return 0
    return int(float(s))


def build_pairs(records, arm_a, arm_b, block_col):
    """Return list of (task_id, block_key, success_a, success_b)."""
    by_task = defaultdict(dict)
    meta = {}
    for r in records:
        lang = str(r["language"]).strip()
        tid = str(r["task_id"]).strip()
        by_task[tid][lang] = to_int(r["success"])
        if tid not in meta:
            meta[tid] = r

    pairs = []
    missing = 0
    for tid, d in by_task.items():
        if arm_a not in d or arm_b not in d:
            missing += 1
            continue
        row = meta[tid]
        if block_col and block_col in row and str(row[block_col]).strip() != "":
            key = f"{block_col}={row[block_col]}"
        else:
            key = f"task={tid}"          # degenerate fallback
        pairs.append((tid, key, d[arm_a], d[arm_b]))

    if missing:
        print(f"[warn] {missing} task(s) lacked one of the two arms; skipped.",
              file=sys.stderr)
    return sorted(pairs)


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------

def discordance(pairs):
    a = b = c = d = 0
    for _, _, sa, sb in pairs:
        if sa and sb:
            a += 1
        elif sb and not sa:
            b += 1          # arm_b solves, arm_a does not
        elif sa and not sb:
            c += 1
        else:
            d += 1
    return a, b, c, d


def exact_mcnemar(b, c):
    """Two-sided exact (binomial) McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def task_permutation(pairs, n_perm, rng):
    """Sign-flip each discordant TASK independently (the paper's current test)."""
    signs = np.array([sb - sa for _, _, sa, sb in pairs if sa != sb], dtype=float)
    if signs.size == 0:
        return 1.0, 0.0
    obs = abs(signs.sum())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, signs.size))
    null = np.abs((flips * np.abs(signs)).sum(axis=1))
    p = (1.0 + (null >= obs - 1e-12).sum()) / (n_perm + 1.0)
    return float(p), float(obs)


def block_permutation(pairs, n_perm, rng):
    """
    Sign-flip whole BLOCKS jointly.

    Under Proposition 2 every task in a block receives the same instruction at
    the deciding step, so the block -- not the task -- is the unit that could
    independently have come out the other way. Flipping blocks jointly is the
    permutation that respects that dependence.
    """
    per_block = defaultdict(float)
    for _, key, sa, sb in pairs:
        if sa != sb:
            per_block[key] += (sb - sa)
    if not per_block:
        return 1.0, 0, 0.0
    vals = np.array(list(per_block.values()), dtype=float)
    obs = abs(vals.sum())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, vals.size))
    null = np.abs((flips * np.abs(vals)).sum(axis=1))
    p = (1.0 + (null >= obs - 1e-12).sum()) / (n_perm + 1.0)
    return float(p), int(vals.size), float(obs)


def exact_block_sign_p(n_discordant_blocks):
    """
    Smallest attainable two-sided sign-test p when every discordant block points
    the same way. This is the honest ceiling on evidence the design can produce.
    """
    if n_discordant_blocks == 0:
        return 1.0
    return min(1.0, 2.0 * (0.5 ** n_discordant_blocks))


def conditional_exact_interval(b, c, n, alpha=0.05):
    """
    Exact interval for S(arm_b) - S(arm_a) CONDITIONAL on b + c discordant pairs,
    obtained by transforming the Clopper-Pearson interval for
    pi = P(a discordance favours arm_b).

        diff = ((b + c) / n) * (2 * pi - 1)

    This is the construction behind the [+0.700, +0.875] in the paper. It is a
    correct interval for the conditional quantity; it is NOT an unconditional
    interval for the difference over task samples, and the paper should say so.
    """
    from scipy.stats import beta
    m = b + c
    if m == 0:
        return (0.0, 0.0), (0.0, 1.0)
    lo_pi = 0.0 if b == 0 else beta.ppf(alpha / 2, b, m - b + 1)
    hi_pi = 1.0 if b == m else beta.ppf(1 - alpha / 2, b + 1, m - b)
    scale = m / n
    return (scale * (2 * lo_pi - 1), scale * (2 * hi_pi - 1)), (lo_pi, hi_pi)


def cluster_bootstrap(pairs, n_boot, rng, alpha=0.05):
    """
    Resample BLOCKS with replacement and recompute the difference. This is the
    unconditional interval the conditional one above is often mistaken for, and
    it is the appropriate interval to quote for the paired difference.
    """
    blocks = defaultdict(list)
    for tid, key, sa, sb in pairs:
        blocks[key].append((sa, sb))
    keys = list(blocks)
    if len(keys) < 2:
        return None
    diffs = []
    for _ in range(n_boot):
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        sa_sum = sb_sum = n = 0
        for i in drawn:
            for sa, sb in blocks[keys[i]]:
                sa_sum += sa
                sb_sum += sb
                n += 1
        diffs.append((sb_sum - sa_sum) / n)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def design_effect(pairs):
    """
    Variance inflation from clustering: how much the blocked structure widens
    the sampling variance of S relative to the binomial assumption. Reported so
    the limitation can be stated as a number rather than as a caveat.
    """
    blocks = defaultdict(list)
    for _, key, _, sb in pairs:
        blocks[key].append(sb)
    sizes = np.array([len(v) for v in blocks.values()], dtype=float)
    if sizes.size < 2:
        return float("nan")
    outcomes = [np.mean(v) for v in blocks.values()]
    p = float(np.mean([x for v in blocks.values() for x in v]))
    if p in (0.0, 1.0):
        return float("nan")
    m_bar = sizes.mean()
    between = np.var(outcomes, ddof=1)
    rho = max(0.0, (between - p * (1 - p) / m_bar) / (p * (1 - p)))
    return float(1 + (m_bar - 1) * rho)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records")
    ap.add_argument("--arm-a", default="L0", help="baseline arm (default L0)")
    ap.add_argument("--arm-b", default="L1", help="treatment arm (default L1)")
    ap.add_argument("--block-col", default=None,
                    help="column holding the decision-context key "
                         "(try 'block', else 'requires_message')")
    ap.add_argument("--n-perm", type=int, default=200_000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    records = load_records(args.records)

    block_col = args.block_col
    if block_col is None:
        cols = set(records[0]) if records else set()
        for cand in ("block", "decision_context", "requires_message"):
            if cand in cols:
                block_col = cand
                break

    pairs = build_pairs(records, args.arm_a, args.arm_b, block_col)
    if not pairs:
        sys.exit("no paired tasks found -- check --arm-a / --arm-b names")

    n = len(pairs)
    a, b, c, d = discordance(pairs)
    diff = (b - c) / n

    p_mcnemar = exact_mcnemar(b, c)
    p_task, _ = task_permutation(pairs, args.n_perm, rng)
    p_block, n_blocks_disc, _ = block_permutation(pairs, args.n_perm, rng)
    p_floor = exact_block_sign_p(n_blocks_disc)
    (ci_lo, ci_hi), (pi_lo, pi_hi) = conditional_exact_interval(b, c, n)
    boot = cluster_bootstrap(pairs, args.n_boot, rng)
    deff = design_effect(pairs)
    n_blocks = len({k for _, k, _, _ in pairs})

    degenerate = block_col is None or all(
        k.startswith("task=") for _, k, _, _ in pairs)

    print("=" * 68)
    print(f"paired comparison: {args.arm_b} vs {args.arm_a}     n = {n} tasks")
    print("=" * 68)
    print(f"discordance table   a={a}  b={b}  c={c}  d={d}")
    print(f"observed difference S({args.arm_b}) - S({args.arm_a}) "
          f"= {b}/{n} - {c}/{n} = {diff:+.3f}")
    print()
    print("-- assuming exchangeable pairs (what the paper currently reports) --")
    print(f"  exact McNemar                       p = {p_mcnemar:.3g}")
    print(f"  task-level sign-flip permutation    p = {p_task:.3g}")
    print(f"  exact CONDITIONAL interval on the difference, given b+c={b+c}:")
    print(f"                                      [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"      (from Clopper-Pearson [{pi_lo:.3f}, {pi_hi:.3f}] on "
          f"pi = P(discordance favours {args.arm_b}))")
    print()
    print("-- respecting the block structure of Proposition 2 --")
    print(f"  block key column                    {block_col or '(none)'}")
    print(f"  blocks                              {n_blocks}  "
          f"(discordant: {n_blocks_disc})")
    print(f"  block-level permutation             p = {p_block:.3g}")
    print(f"  smallest attainable two-sided p     {p_floor:.3g}"
          f"   <- evidence ceiling of this design")
    if boot:
        print(f"  cluster bootstrap 95% CI on diff    "
              f"[{boot[0]:+.3f}, {boot[1]:+.3f}]")
    if not math.isnan(deff):
        print(f"  design effect (variance inflation)  {deff:.2f}x")
    print()

    if degenerate:
        print("!! No block column was found, so every task was treated as its own")
        print("!! block and the numbers above just reproduce the paper. Add a")
        print("!! 'block' column (the argmax-invariant prefix at the deciding")
        print("!! step) or 'requires_message' and re-run -- that is the point.")
    else:
        print("Report the block-level p and the number of discordant blocks next")
        print("to the exact McNemar value, and state that the exact value assumes")
        print("an exchangeability the design violates. If the block-level p is")
        print("weak, that is not a weaker result: the finding is the COMPLETE")
        print("separation (c = 0) plus its reproduction across five task samples.")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({
                "n": n, "a": a, "b": b, "c": c, "d": d, "difference": diff,
                "p_exact_mcnemar": p_mcnemar,
                "p_task_permutation": p_task,
                "p_block_permutation": p_block,
                "p_attainable_floor": p_floor,
                "n_blocks": n_blocks, "n_blocks_discordant": n_blocks_disc,
                "conditional_interval": [ci_lo, ci_hi],
                "cluster_bootstrap_interval": boot,
                "design_effect": deff,
                "block_col": block_col,
            }, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

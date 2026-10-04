#!/usr/bin/env python3
"""
extra_analysis.py -- statistics and LaTeX rows for the extra runs.

    python extra_analysis.py results/extra                 # everything found
    python extra_analysis.py results/extra --predictions predictions.json
    python extra_analysis.py --demo                        # synthetic records, checks the code path

Per (model, seed): S on seen and counterfactual tasks, Htot, ordering-cf
success, the L1-vs-L0 discordance table (b, c, a, d), exact two-sided
McNemar, a sign-flip permutation p (100,000 resamples, seed 0), and the exact
Clopper-Pearson interval on the fraction of tasks gained (b/n).  Across seeds:
mean, min, whether cell c is empty on every seed, and the smallest per-seed
gap -- the quantities the paper reports for seeds 0-4.

Block-aware inference (cluster bootstrap over decision blocks) is NOT
recomputed here; it needs the block ids from your sandbox. Run your
analysis.py on the same JSONL for that.

Conventions follow the paper: b = tasks L1 solves and L0 does not; c = the
reverse; a = both solve; d = neither. Htot in nats. "Within 0.10" is the
paper's prediction-hit criterion.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

LANG_TEX = {"L0": r"$\Lang_0$", "L0p": r"$\Lang_0'$", "L1": r"$\Lang_1$",
            "L2": r"$\Lang_2$", "L3": r"$\Lang_3$", "Lrand": r"$\Lang_{\rm rand}$"}
MODEL_SHORT = {"Qwen/Qwen3-0.6B": "0.6B", "Qwen/Qwen3-1.7B": "1.7B", "Qwen/Qwen3-4B": "4B"}


# ----------------------------------------------------------------------------- io
def load(root: Path) -> dict:
    """{(model, seed, lang): [records]}"""
    cells = defaultdict(list)
    for p in sorted(root.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cells[(r["model"], int(r["seed"]), r["language"])].append(r)
    return cells


# ----------------------------------------------------------------------------- stats
def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * binom_cdf(k, n, 0.5))


def signflip_p(diffs: list[int], n_resample: int = 100_000, seed: int = 0) -> float:
    """Two-sided permutation p for the mean paired difference under sign flips."""
    rng = random.Random(seed)
    obs = abs(sum(diffs))
    nz = [d for d in diffs if d != 0]
    if not nz:
        return 1.0
    hits = 0
    for _ in range(n_resample):
        s = sum(d if rng.random() < 0.5 else -d for d in nz)
        if abs(s) >= obs:
            hits += 1
    return max(hits, 1) / n_resample if hits == 0 else hits / n_resample


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact interval for a binomial proportion by bisection on the binomial CDF."""
    if n == 0:
        return (0.0, 1.0)
    def solve(target, lo, hi, f):
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(alpha / 2, 0.0, 1.0, lambda p: 1 - binom_cdf(k - 1, n, p))
    # for the lower bound we need P(X >= k | p) = alpha/2, increasing in p
    if k > 0:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if 1 - binom_cdf(k - 1, n, mid) < alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    upper = 1.0 if k == n else None
    if k < n:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if binom_cdf(k, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return (lower, upper)


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx)**2 for a in rx) * sum((b - my)**2 for b in ry))
    return num / den if den else float("nan")


# ----------------------------------------------------------------------------- cell summaries
def summarize_cell(recs: list[dict]) -> dict:
    seen = [r for r in recs if r["split"] == "seen"]
    cf = [r for r in recs if r["split"] == "cf"]
    ordering = [r for r in cf if r.get("cf_dim") == "ordering"]
    S = lambda rs: sum(r["success"] for r in rs) / len(rs) if rs else float("nan")
    H = lambda rs: sum(r["htot"] for r in rs) / len(rs) if rs else float("nan")
    return dict(n_seen=len(seen), n_cf=len(cf), S=S(seen), S_cf=S(cf), Htot=H(seen),
                Kstruct=sum(r["kstruct"] for r in seen) / len(seen) if seen else float("nan"),
                Kparam=sum(r["kparam"] for r in seen) / len(seen) if seen else float("nan"),
                ordering_cf=S(ordering), illegal=sum(r["halted_illegal"] for r in seen))


def paired(cell_a: list[dict], cell_b: list[dict], split="seen") -> dict:
    """Discordance of B over A on `split` (B = the arm expected to win)."""
    A = {r["task_id"]: r["success"] for r in cell_a if r["split"] == split}
    B = {r["task_id"]: r["success"] for r in cell_b if r["split"] == split}
    ids = sorted(set(A) & set(B))
    b = sum(1 for i in ids if B[i] and not A[i])
    c = sum(1 for i in ids if A[i] and not B[i])
    a = sum(1 for i in ids if A[i] and B[i])
    d = sum(1 for i in ids if not A[i] and not B[i])
    diffs = [int(B[i]) - int(A[i]) for i in ids]
    lo, hi = clopper_pearson(b, len(ids))
    return dict(n=len(ids), b=b, c=c, a=a, d=d, gap=(b - c) / len(ids) if ids else float("nan"),
                p_mcnemar=mcnemar_exact(b, c), p_signflip=signflip_p(diffs),
                cp_gain=(lo, hi))


# ----------------------------------------------------------------------------- report
def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def report(cells: dict, predictions: dict | None):
    models = sorted({m for m, _, _ in cells})
    lines = []
    tex_ladder, tex_family = [], []
    for m in models:
        seeds = sorted({s for mm, s, _ in cells if mm == m})
        langs = [l for l in ["L0", "L0p", "L1", "L2", "L3", "Lrand"] if any((m, s, l) in cells for s in seeds)]
        short = MODEL_SHORT.get(m, m.split("/")[-1])
        lines.append(f"\n=== {m}  seeds {seeds}  languages {langs}")
        lines.append(f"{'seed':>4} {'lang':>5} {'n':>3} {'S':>6} {'S_cf':>6} {'Htot':>6} {'Ks':>5} {'Kp':>6} {'ord':>5} {'illegal':>7}")
        per_lang = defaultdict(list)
        pairs = []
        for s in seeds:
            summ = {}
            for l in langs:
                if (m, s, l) not in cells:
                    continue
                summ[l] = summarize_cell(cells[(m, s, l)])
                q = summ[l]
                per_lang[l].append(q)
                lines.append(f"{s:>4} {l:>5} {q['n_seen']:>3} {fmt(q['S']):>6} {fmt(q['S_cf']):>6} {fmt(q['Htot']):>6} "
                             f"{fmt(q['Kstruct'],2):>5} {fmt(q['Kparam'],2):>6} {fmt(q['ordering_cf'],2):>5} {q['illegal']:>7}")
                tex_ladder.append(f"{short} & {LANG_TEX.get(l, l)} & {fmt(q['S'])} & {fmt(q['S_cf'])} & "
                                  f"{fmt(q['Htot'])} & {fmt(q['ordering_cf'], 2)} \\\\  % seed {s}")
            if "L0" in summ and "L1" in summ:
                pr = paired(cells[(m, s, "L0")], cells[(m, s, "L1")])
                pairs.append(pr)
                lines.append(f"     L1 vs L0 seen: b={pr['b']} c={pr['c']} a={pr['a']} d={pr['d']}  gap={fmt(pr['gap'])}  "
                             f"McNemar p={pr['p_mcnemar']:.2e}  sign-flip p={pr['p_signflip']:.1e}  "
                             f"CP(gain)=[{fmt(pr['cp_gain'][0])}, {fmt(pr['cp_gain'][1])}]")
            if "L0p" in summ and "L1" in summ:
                pr2 = paired(cells[(m, s, "L0p")], cells[(m, s, "L1")])
                lines.append(f"     L1 vs L0': b={pr2['b']} c={pr2['c']}  McNemar p={pr2['p_mcnemar']:.2e}")
        # across seeds
        if len(seeds) > 1:
            lines.append("  -- across seeds --")
            for l in langs:
                Ss = [q["S"] for q in per_lang[l]]
                if Ss:
                    lines.append(f"     {l:>4}: mean S={fmt(sum(Ss)/len(Ss))}  min={fmt(min(Ss))}  max={fmt(max(Ss))}  "
                                 f"per-seed={[round(x,3) for x in Ss]}")
                    tex_family.append(f"{short} & {LANG_TEX.get(l, l)} & {fmt(sum(Ss)/len(Ss))} & "
                                      f"{fmt(min(Ss))}--{fmt(max(Ss))} & {len(Ss)} \\\\")
            if pairs:
                ranking_ok = all(
                    (lambda q: q.get("L1", float("nan")) > q.get("L0p", -1) > q.get("L0", -2))(
                        {l: per_lang[l][i]["S"] for l in langs if i < len(per_lang[l])})
                    for i in range(len(seeds))) if "L0p" in langs else None
                lines.append(f"     cell c empty on every seed: {all(p['c'] == 0 for p in pairs)}  "
                             f"(c per seed {[p['c'] for p in pairs]})")
                lines.append(f"     smallest per-seed gap L1-L0: {fmt(min(p['gap'] for p in pairs))}")
                if ranking_ok is not None:
                    lines.append(f"     L1 > L0' > L0 on every seed: {ranking_ok}")
            # entropy vs success across all (seed, lang) cells of this model
            xs = [q["Htot"] for l in langs for q in per_lang[l]]
            ys = [q["S"] for l in langs for q in per_lang[l]]
            if len(xs) >= 4:
                lines.append(f"     Spearman rho(Htot, S) over {len(xs)} cells: {spearman(xs, ys):+.2f}")
        # predictions
        if predictions and m in predictions:
            lines.append("  -- documented predictions --")
            for key, pred in predictions[m].items():
                lang, field = key.split(".")
                vals = [q[field] for q in per_lang.get(lang, []) if field in q]
                if not vals:
                    lines.append(f"     {key}: predicted {pred}, no data")
                    continue
                obs = sum(vals) / len(vals)
                hit = abs(obs - pred) <= 0.10
                lines.append(f"     {key}: predicted {pred:.3f}  observed {obs:.3f}  {'HIT' if hit else 'MISS'} (|d|={abs(obs-pred):.3f})")
    print("\n".join(lines))
    print("\n=== LaTeX rows, ladder-table format (Model & Language & S & S_cf & Htot & ordering cf) ===")
    print("\n".join(tex_ladder))
    if tex_family:
        print("\n=== LaTeX rows, across-seed format (Model & Language & mean S & range & seeds) ===")
        print("\n".join(tex_family))


# ----------------------------------------------------------------------------- demo
def _demo(root: Path):
    """Synthetic records with the paper's qualitative shape, to exercise the code."""
    rng = random.Random(0)
    root.mkdir(parents=True, exist_ok=True)
    for model, base in [("Demo/Family-360M", {"L0": 0.0, "L0p": 0.15, "L1": 0.8}),
                        ("Demo/Family-1.7B", {"L0": 0.2, "L0p": 0.4, "L1": 0.5})]:
        for seed in range(5):
            for lang, p in base.items():
                path = root / model.replace("/", "__") / f"seed{seed}" / f"{lang}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    for i in range(70):
                        split = "seen" if i < 40 else "cf"
                        succ = rng.random() < p
                        f.write(json.dumps(dict(model=model, revision="demo", seed=seed, language=lang,
                                                task_id=f"t{i}", split=split,
                                                cf_dim=None if split == "seen" else ["channel", "side_effect", "ordering", "primitive_sub", "data_dependency"][(i - 40) // 6],
                                                source_id=None if split == "seen" else f"t{i-40}",
                                                success=succ, htot=rng.gauss(3.4 if lang == "L0" else 2.6, 0.2),
                                                kstruct=rng.choice([2, 3, 4]), kparam=rng.choice([8, 9, 10]),
                                                n_instr=3, halted_illegal=(not succ and rng.random() < 0.8), history=[])) + "\n")
    print(f"demo records written to {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/extra")
    ap.add_argument("--predictions", help="json: {model_id: {'L1.S': 0.8, 'L0.S': 0.0, ...}}")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if a.demo:
        root = Path("results/_demo"); _demo(root)
    cells = load(root)
    if not cells:
        sys.exit(f"no records under {root}")
    preds = json.loads(Path(a.predictions).read_text()) if a.predictions else None
    report(cells, preds)


if __name__ == "__main__":
    main()

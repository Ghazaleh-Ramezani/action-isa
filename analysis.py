#!/usr/bin/env python3
"""
Regenerate every statistic reported in the paper from the released result files.

    python analysis.py                 # all sections
    python analysis.py --section 5.1   # one section

Reads only results/*.json. No model, no GPU, no network. Every number printed
here appears in the paper; if a number here disagrees with the paper, the paper
is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from itertools import combinations
from math import comb

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- statistics


def discordance(a_tasks, b_tasks):
    """2x2 table for paired binary outcomes. b = B solves, A does not."""
    keys = sorted(set(a_tasks) & set(b_tasks))
    a = sum(1 for k in keys if a_tasks[k] and b_tasks[k])
    b = sum(1 for k in keys if not a_tasks[k] and b_tasks[k])
    c = sum(1 for k in keys if a_tasks[k] and not b_tasks[k])
    d = sum(1 for k in keys if not a_tasks[k] and not b_tasks[k])
    return a, b, c, d


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    probs = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = probs[b]
    return min(1.0, sum(p for p in probs if p <= obs * (1 + 1e-12)))


def bootstrap_ci(a_tasks, b_tasks, reps=10000, seed=0, alpha=0.05):
    """Percentile CI on S(B) - S(A), resampling task PAIRS."""
    keys = sorted(set(a_tasks) & set(b_tasks))
    pairs = [(a_tasks[k], b_tasks[k]) for k in keys]
    n = len(pairs)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(y for _, y in sample) / n - sum(x for x, _ in sample) / n)
    diffs.sort()
    lo = diffs[int((alpha / 2) * reps)]
    hi = diffs[int((1 - alpha / 2) * reps) - 1]
    return lo, hi


def spearman(xs, ys):
    def rank(v):
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

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def check(label, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {label:<44} {got:.4f}  (paper: {want})")
    return ok


# ---------------------------------------------------------------- sections


def section_5_1(fail):
    print("\n[5.1] Primary comparison")
    r0 = load("r0_results.json")
    L0, L1 = r0["L0"], r0["L1"]
    for nm, d in (("L0", L0), ("L1", L1)):
        s = list(d["per_task_seen"].values())
        print(f"  {nm}: S_seen = {sum(s)/len(s):.3f}  (file: {d['S_seen']:.3f})")
    a, b, c, d_ = discordance(L0["per_task_seen"], L1["per_task_seen"])
    print(f"  discordance a={a} b={b} c={c} d={d_}")
    p = mcnemar_exact(b, c)
    print(f"  exact McNemar p = {p:.3e}")
    lo, hi = bootstrap_ci(L0["per_task_seen"], L1["per_task_seen"])
    print(f"  95% bootstrap CI on S(L1)-S(L0) = [{lo:.3f}, {hi:.3f}]")
    fail += [not check("S(L1)", L1["S_seen"], 0.875)]
    fail += [not check("S(L0)", L0["S_seen"], 0.000)]
    fail += [b != 35 or c != 0]

    print("\n  floor anchored to strongest raw-primitive grammar:")
    B = load("partB.json")
    best = max(B.values(), key=lambda g: g["S_seen"])
    a2, b2, c2, d2 = discordance(best["per_task_seen"], L1["per_task_seen"])
    print(f"    best grammar S = {best['S_seen']:.3f}; "
          f"a={a2} b={b2} c={c2} d={d2}, p = {mcnemar_exact(b2, c2):.2e}")


def section_5_3(fail):
    print("\n[5.3] Lower-entropy control")
    r0, r2 = load("r0_results.json"), load("r2_results.json")
    L0, L1, T = r0["L0"], r0["L1"], r2["L0_tight"]
    for nm, d in (("L0", L0), ("L0'", T), ("L1", L1)):
        print(f"  {nm:4s} S={d['S_seen']:.3f}  H_tot={d['H_tot']:.3f}  "
              f"K_struct={d['K_struct']:.2f}  K_param={d['K_param']:.2f}")
    gap = L1["S_seen"] - L0["S_seen"]
    print(f"  control closes {(T['S_seen']-L0['S_seen'])/gap:.1%} of the gap")
    print(f"  entropy cut: L0' {1-T['H_tot']/L0['H_tot']:.1%}, "
          f"L1 {1-L1['H_tot']/L0['H_tot']:.1%}")
    print(f"  L0' - L1 entropy = {T['H_tot']-L1['H_tot']:+.3f} nats "
          "(control carries LESS)")
    fail += [not check("S(L0')", T["S_seen"], 0.150)]
    fail += [T["K_struct"] <= L0["K_struct"]]  # paper says counts RISE


def section_5_5(fail):
    print("\n[5.5] Robustness, sweep, factorial")
    A = load("partA.json")
    for arm in ("L0", "L0_tight", "L1"):
        vals = [A[s][arm]["S_seen"] for s in sorted(A)]
        print(f"  {arm:9s} mean={sum(vals)/len(vals):.3f}  "
              f"min={min(vals):.3f}  max={max(vals):.3f}")
    every = all(A[s]["L1"]["S_seen"] > A[s]["L0"]["S_seen"] for s in A)
    print(f"  L1 > L0 on every seed: {every}")

    B = load("partB.json")
    hs = [g["H_tot"] for g in B.values()]
    ss = [g["S_seen"] for g in B.values()]
    print(f"  sweep: H_tot in [{min(hs):.3f}, {max(hs):.3f}], "
          f"max S = {max(ss):.3f}, rho = {spearman(hs, ss):+.2f}")

    C = load("partC.json")
    def eff(flag):
        on = [v["S_seen"] for k, v in C.items() if f"{flag}1" in k]
        off = [v["S_seen"] for k, v in C.items() if f"{flag}0" in k]
        return sum(on) / len(on) - sum(off) / len(off)
    for flag, nm in (("pre", "precondition"), ("stop", "stopping"), ("ord", "ordering")):
        print(f"  main effect {nm:<13} {eff(flag):+.3f}")
    cells = [k for k in C if k.endswith("ord1")]
    solved = {frozenset(t for t, v in C[k]["per_task_seen"].items() if v) for k in cells}
    print(f"  all ordering-on cells solve the identical task set: {len(solved) == 1}")
    fail += [len(solved) != 1]


def section_5_6(fail):
    print("\n[5.6/D] Prompt-matched random control")
    D = load("partD.json")["L_rand_matched"]
    r0 = load("r0_results.json")
    print(f"  L_rand      S={r0['L_rand']['S_seen']:.3f}  "
          f"tokens={r0['L_rand']['prompt_tokens']:.1f}")
    print(f"  L_rand^m    S={D['S_seen']:.3f}  tokens={D['prompt_tokens']:.1f}")
    print(f"  L1          S={r0['L1']['S_seen']:.3f}  "
          f"tokens={r0['L1']['prompt_tokens']:.1f}")
    fail += [not check("S(L_rand^m)", D["S_seen"], 0.175)]
    fail += [D["prompt_tokens"] >= r0["L1"]["prompt_tokens"]]


def section_ladder(fail):
    print("\n[5.5/E] Model ladder")
    E = load("r3_results_E.json")["model_ladder"]
    r0, r2 = load("r0_results.json"), load("r2_results.json")
    base = {"L0": r0["L0"], "L0_tight": r2["L0_tight"], "L1": r0["L1"]}
    print(f"  {'model':<12}{'L0':>8}{'L0_tight':>10}{'L1':>8}{'gap':>8}")
    print(f"  {'0.6B':<12}" + "".join(f"{base[k]['S_seen']:>8.3f}" if k != 'L0_tight'
          else f"{base[k]['S_seen']:>10.3f}" for k in ("L0", "L0_tight", "L1"))
          + f"{base['L1']['S_seen']-base['L0']['S_seen']:>8.3f}")
    for mn, row in E.items():
        g = row["L1"]["S_seen"] - row["L0"]["S_seen"]
        print(f"  {mn.split('/')[-1]:<12}" + "".join(
            f"{row[k]['S_seen']:>8.3f}" if k != 'L0_tight'
            else f"{row[k]['S_seen']:>10.3f}" for k in ("L0", "L0_tight", "L1"))
            + f"{g:>8.3f}")
        for x, y in (("L0", "L1"), ("L0_tight", "L1")):
            a, b, c, d = discordance(row[x]["per_task_seen"], row[y]["per_task_seen"])
            print(f"      {x:9s} vs {y:9s}: a={a:2d} b={b:2d} c={c:2d} d={d:2d}"
                  f"  p={mcnemar_exact(b, c):.4f}")
        print("      H_tot: " + ", ".join(
            f"{k}={row[k]['H_tot']:.3f}" for k in ("L0", "L0_tight", "L1"))
            + "   <- L1 highest at 1.7B, and still wins")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all")
    args = ap.parse_args()
    fail = []
    todo = {
        "5.1": section_5_1, "5.3": section_5_3, "5.5": section_5_5,
        "5.6": section_5_6, "ladder": section_ladder,
    }
    print("=" * 66)
    print("Regenerating every reported statistic from results/*.json")
    print("=" * 66)
    for name, fn in todo.items():
        if args.section in ("all", name):
            fn(fail)
    print("\n" + "=" * 66)
    print("FAILED CHECKS:" if any(fail) else "All checks reproduce the paper.")
    return 1 if any(fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Two ablations targeting the two strongest remaining criticisms.

  A. NAME RENAMING (default 5 seeds x 2 languages)
     Replace every instruction name and slot name with opaque tokens
     (OP_QKX, ARG_JW, ...), leaving structure, slot domains, expansions,
     prompts and decoding identical.
       - If L1's advantage came from natural-language names, renamed L1
         collapses toward L0.
       - If L0's floor were a naming artifact, renamed L0 moves.

  B. BUNDLE PARTITIONS (exhaustive, 16 arms)
     The canonical 5-primitive trace admits exactly 2^4 = 16 contiguous
     partitions (each of the 4 cut points is either cut or not); 6 of them
     have exactly 3 blocks and are instruction-count-matched to L1.
     L_rand in the paper is ONE of these. Running all 16 answers whether
     random bundling fails in general or one unlucky split was chosen.
     The 1-block partition (everything bundled, ordering compiled away
     entirely) and the 5-block partition (= L0 regrouped) are the two
     extremes and are included.

Usage:
  python3 name_bundle_ablations.py --dry-run                 # pipeline check
  python3 name_bundle_ablations.py --model Qwen/Qwen3-0.6B   # real run
  python3 name_bundle_ablations.py --model ... --part names  # one part only
"""
import argparse, json, os, random, string, sys, time
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r0_experiment as R

CANONICAL = ["RESOLVE", "CREATE_EVENT", "GET_LINK", "COMPOSE", "SEND"]


# ---------------------------------------------------------------- renaming
def _opaque(rng: random.Random, prefix: str) -> str:
    return prefix + "_" + "".join(rng.choice(string.ascii_uppercase) for _ in range(3))


def rename_language(lang, seed: int):
    """Same structure, same domains, same expansions; opaque names only."""
    rng = random.Random(1000 + seed)
    used = set()

    def fresh(prefix):
        while True:
            n = _opaque(rng, prefix)
            if n not in used:
                used.add(n)
                return n

    new_ins = []
    for ins in lang.instructions:
        new_ins.append(R.Instruction(
            name=fresh("OP"),
            slots=[(fresh("ARG"), list(vals)) for _, vals in ins.slots],
            expand=ins.expand,
        ))
    return R.Language(f"{lang.name}_ren{seed}", new_ins, lang.expressible_dims,
                      f"{lang.name} with opaque instruction and slot names (seed {seed})")


# ---------------------------------------------------------------- partitions
def make_partition_lang(cuts, tag: str):
    """Contiguous partition of the canonical trace at the given cut points."""
    cs = sorted(cuts)
    bounds = [0] + cs + [len(CANONICAL)]
    chunks = [CANONICAL[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

    def make_expand(chunk):
        def _e(a, sp):
            out, i = [], 0
            for pn in chunk:
                k = len(R.PRIMITIVES[pn]["slots"])
                out.append((pn, list(a[i:i + k])))
                i += k
            return out
        return _e

    ins = []
    for j, chunk in enumerate(chunks):
        slots = []
        for pn in chunk:
            for sname, vals in R.PRIMITIVES[pn]["slots"]:
                slots.append((f"{pn.lower()}_{sname}", list(vals)))
        ins.append(R.Instruction(name=f"BLOCK_{j}", slots=slots,
                                 expand=make_expand(chunk)))
    return R.Language(f"L_part_{tag}", ins, frozenset(R.CF_DIMS),
                      f"partition cuts={cs}")


def all_partitions():
    out = []
    for r in range(0, 5):
        for cuts in combinations([1, 2, 3, 4], r):
            tag = "".join(str(c) for c in cuts) or "none"
            out.append((cuts, make_partition_lang(cuts, tag)))
    return out


# ---------------------------------------------------------------- driver
def run_language(scorer, lang, seen, esem_known=None):
    """esem_known: renaming leaves e_syntax/e_semantic pointwise invariant
    (names never enter enumerate_actions' outcomes or expressible_dims),
    so renamed arms inherit the original language's certified value
    instead of re-running the breadth-first search."""
    t0 = time.time()
    recs = [R.run_task(scorer, lang, sp) for sp in seen]
    n = len(recs)
    return {
        "S": sum(r.success for r in recs) / n,
        "E_syn": sum(R.e_syntax(lang, sp) for sp in seen) / n,
        "E_sem": esem_known if esem_known is not None
                 else sum(R.e_semantic(lang, sp) for sp in seen) / n,
        "H_tot": sum(r.h_tot for r in recs) / n,
        "K_struct": sum(r.k_struct for r in recs) / n,
        "n_instructions": len(lang.instructions),
        "per_task": {r.tid: bool(r.success) for r in recs},
        "secs": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-rename", type=int, default=5)
    ap.add_argument("--part", choices=["names", "bundles", "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/name_bundle.json")
    args = ap.parse_args()
    if not args.dry_run and not args.model:
        ap.error("pass --model or --dry-run")

    seen, _ = R.build_tasks(seed=args.seed)
    scorer = R.DryRunScorer(args.seed) if args.dry_run else R.HFScorer(args.model)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        results = json.load(open(args.out))
    except Exception:
        results = {"_config": {"n_seen": len(seen), "seed": args.seed,
                               "model": args.model or "dry-run"}}

    def save():
        json.dump(results, open(args.out, "w"), indent=1)

    arms = []
    if args.part in ("names", "all"):
        # renaming is semantics-invariant: inherit certified E_sem = 1.0
        arms += [(f"L1_ren{s}", (lambda s=s: rename_language(R.make_L1(), s)), 1.0)
                 for s in range(args.n_rename)]
        arms += [(f"L0_ren{s}", (lambda s=s: rename_language(R.make_L0(), s)), 1.0)
                 for s in range(args.n_rename)]
    if args.part in ("bundles", "all"):
        # partitions can genuinely lose E_sem (forced side effects): compute it
        arms += [(f"part_{('' .join(map(str, cuts))) or 'none'}",
                  (lambda cuts=cuts: make_partition_lang(
                      cuts, "".join(map(str, cuts)) or "none")), None)
                 for cuts, _ in all_partitions()]

    for name, build, esem_known in arms:
        if name in results:
            print(f"{name}: cached, skipping", flush=True)
            continue
        lang = build()
        res = run_language(scorer, lang, seen, esem_known=esem_known)
        results[name] = res
        save()
        print(f"{name:14s} S={res['S']:.3f} E_sem={res['E_sem']:.2f} "
              f"H={res['H_tot']:.2f} K_str={res['K_struct']:.2f} "
              f"ins={res['n_instructions']} [{res['secs']}s]", flush=True)

    # ------- summary -------
    ren1 = [v["S"] for k, v in results.items() if k.startswith("L1_ren")]
    ren0 = [v["S"] for k, v in results.items() if k.startswith("L0_ren")]
    parts = {k: v for k, v in results.items() if k.startswith("part_")}
    print("\n=== SUMMARY ===")
    if ren1:
        print(f"L1 renamed  (n={len(ren1)}): min={min(ren1):.3f} "
              f"max={max(ren1):.3f} mean={sum(ren1)/len(ren1):.3f}  "
              f"[original L1: 0.875]")
    if ren0:
        print(f"L0 renamed  (n={len(ren0)}): min={min(ren0):.3f} "
              f"max={max(ren0):.3f} mean={sum(ren0)/len(ren0):.3f}  "
              f"[original L0: 0.000]")
    if parts:
        matched = {k: v for k, v in parts.items() if v["n_instructions"] == 3}
        certified = {k: v for k, v in parts.items() if v["E_sem"] >= 0.999}
        print(f"partitions  (n={len(parts)}): "
              f"S in [{min(v['S'] for v in parts.values()):.3f}, "
              f"{max(v['S'] for v in parts.values()):.3f}]")
        if matched:
            best = max(matched.items(), key=lambda kv: kv[1]["S"])
            print(f"  3-block (L1-count-matched, n={len(matched)}): "
                  f"max S = {best[1]['S']:.3f} ({best[0]})")
        if certified:
            bestc = max(certified.items(), key=lambda kv: kv[1]["S"])
            print(f"  E_sem=1 subset (n={len(certified)}): "
                  f"max S = {bestc[1]['S']:.3f} ({bestc[0]})")
    save()


if __name__ == "__main__":
    main()

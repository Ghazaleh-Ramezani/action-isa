#!/usr/bin/env python3
"""
ablations.py -- driver for the four runs that would most improve the paper.

Everything here is a configuration plus a thin driver. The only thing you need
to supply is the HOOKS block: three functions from your existing harness. The
grammar variants, the scorer variant and the run grid are fully specified below,
so once the hooks are wired every experiment is one command.

    python ablations.py scorer      --out results/scorer/
    python ablations.py granularity --out results/granularity/
    python ablations.py randseeds   --out results/randseeds/
    python ablations.py family      --out results/family/
    python ablations.py ladder      --out results/ladder/

Each ablation targets one candidate explanation of the primary gap; see the paper's appendix for what each run establishes.
"""

import argparse
import itertools
import json
import os
import sys

# ============================================================================
# HOOKS -- wire these three to your harness, then everything below runs.
# ============================================================================

# All three hooks live in harness_hooks.py, wired to the released harness
# (r0_experiment.py + r3_all_in_one.py). Nothing below needs editing.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

try:
    from harness_hooks import load_grammar, register_grammar, run_arm
except ImportError as _exc:
    raise SystemExit(
        "harness_hooks.py could not be imported: %s\n"
        "It must sit next to ablations.py, together with r0_experiment.py "
        "and r3_all_in_one.py." % _exc
    )


# ============================================================================
# 1. SCORER ABLATION
# ============================================================================
# Why: candidates are ranked by the SUMMED log-probability of their own tokens.
# L1's instruction names and slot sets differ in length from L0's, so a reader
# can ask whether the gap is a length artifact of the scoring rule rather than a
# fact about decision form. The paper matched name-token length for L_rand but
# never for L0 vs L1. Running the same loop under a length-normalised scorer
# either closes that objection outright or finds something you need to know.
#
#   sum_logprob   score(a) = sum_i log p(a_i | prompt, a_<i)        [current]
#   mean_logprob  score(a) = (1/|a|) * sum_i log p(a_i | ...)       [new]
#
# Report both columns for L0, L0', L1 at 0.6B. If the ordering is unchanged, one
# sentence in Section 3 and a row in Table 7 closes the confound permanently.

SCORERS = ["sum_logprob", "mean_logprob"]
SCORER_GRID = dict(
    models=["Qwen/Qwen3-0.6B"],
    grammars=["L0", "L0prime", "L1"],
    task_seeds=[0],
    scorers=SCORERS,
)


# ============================================================================
# 2. BUNDLING-GRANULARITY LADDER
# ============================================================================
# Why: this is the biggest open hole in the paper. The 2x2x2 factorial accounts
# for 0.1375 of a 0.875 gap; the remaining 0.725 is attributed to "bundling" and
# decomposed by nothing. A monotone series of grammars that bundle progressively
# more of the dependency chain, at Esyn = Esem = 1 throughout, turns that black
# box into a curve and shows WHERE success appears.
#
# Chain (Appendix D): RESOLVE -> CREATE_EVENT -> GET_LINK -> COMPOSE -> SEND
# Every removed sequencing distinction is retained as a typed slot, exactly as
# the recipe requires, so expressivity is held at 1 by construction -- but
# certify it with your oracle search anyway, in every rung.

CONTACTS = ["sara", "ali", "maryam", "reza"]
TOPICS   = ["t1", "t2", "t3", "t4"]
TIMES    = ["m1", "m2", "m3", "m4"]
CHANNELS = ["email", "slack"]

GRANULARITY_LADDER = {
    # G0 == L0, five raw primitives (already registered; listed for completeness)
    "G0": None,

    # G1: bundle the first dependency edge only.
    "G1": {"instructions": [
        {"name": "RESOLVE_AND_CREATE",
         "slots": {"contact": CONTACTS, "time": TIMES, "topic": TOPICS}},
        {"name": "GET_LINK",
         "slots": {"topic": TOPICS, "time": TIMES, "attendee": CONTACTS}},
        {"name": "COMPOSE",
         "slots": {"recipient": CONTACTS, "topic": TOPICS,
                   "include_link": [True, False], "channel": CHANNELS}},
        {"name": "SEND",
         "slots": {"recipient": CONTACTS, "topic": TOPICS,
                   "include_link": [True, False], "channel": CHANNELS}},
    ]},

    # G2: bundle the message half as well (the precondition L0 violates).
    "G2": {"instructions": [
        {"name": "RESOLVE_AND_CREATE",
         "slots": {"contact": CONTACTS, "time": TIMES, "topic": TOPICS}},
        {"name": "GET_LINK",
         "slots": {"topic": TOPICS, "time": TIMES, "attendee": CONTACTS}},
        {"name": "COMPOSE_AND_SEND",
         "slots": {"recipient": CONTACTS, "topic": TOPICS,
                   "include_link": [True, False], "channel": CHANNELS}},
    ]},

    # G3: bundle the link into the message, leaving two units and an order.
    "G3": {"instructions": [
        {"name": "RESOLVE_AND_CREATE",
         "slots": {"contact": CONTACTS, "time": TIMES, "topic": TOPICS}},
        {"name": "NOTIFY",
         "slots": {"recipient": CONTACTS, "topic": TOPICS, "time": TIMES,
                   "include_link": [True, False], "channel": CHANNELS}},
    ]},

    # G4 == L1, one unit, ordering demoted to a slot (already registered).
    "G4": None,
}

GRANULARITY_GRID = dict(
    models=["Qwen/Qwen3-0.6B"],
    grammars=["L0", "G1", "G2", "G3", "L1"],
    task_seeds=[0, 1, 2, 3, 4],
    scorers=["sum_logprob"],
)

# What to look for. Two shapes, two different papers:
#   - a STEP at one rung  -> that edge carries the effect; you can name it, and
#     the 0.725 stops being undecomposed
#   - a smooth RAMP       -> the effect is cumulative in structural-decision
#     count, which contradicts the "kind not count" framing and you must say so
# Either way you should also plot K_struct and H_tot per rung: if S tracks
# K_struct rather than the rung index, that is a finding about count.


# ============================================================================
# 3. RANDOM CONTROL, MULTIPLE CUT SEEDS
# ============================================================================
# Why: L_rand currently uses ONE cut seed, and Appendix A already admits its
# 0.10 comes from a lucky cut that happened to trap GET_LINK and COMPOSE in one
# block. A single draw from a random control is not a control. Ten cut seeds
# turn a point estimate into a distribution, and "L1 = 0.875 lies above the 95th
# percentile of random bundling" is a far stronger sentence than "0.175".
# Cheap: no new grammars to design, no new model.

RANDSEED_GRID = dict(
    models=["Qwen/Qwen3-0.6B"],
    grammars=["Lrand_matched"],
    task_seeds=[0],
    cut_seeds=list(range(10)),
    scorers=["sum_logprob"],
)
# Report: median and [min, max] of S over cut seeds, and the
# attainability-normalised version, since Esem varies by cut.


# ============================================================================
# 4. SECOND MODEL FAMILY
# ============================================================================
# Why: this is the single largest external-validity gap, and it is also the only
# thing that can settle the 4B reversal. Right now L1 is non-monotone
# (0.875 -> 0.450 -> 0.400) and the lower-entropy control overtakes it at 4B,
# with no account. That is compatible with the 0.6B result being a property of
# one checkpoint's token preferences. A second family at comparable scale tests
# whether the 0.6B ordering is a property of the recoding or of this
# checkpoint; either outcome is reported (see the second-family appendix).
#
# Pick two of these; they are all small enough to run on one consumer GPU.

FAMILY_GRID = dict(
    models=[
        "Qwen/Qwen3-0.6B",              # anchor, already have it
        "meta-llama/Llama-3.2-1B",
        "HuggingFaceTB/SmolLM2-360M",
        "HuggingFaceTB/SmolLM2-1.7B",
        "google/gemma-3-1b-pt",
    ],
    grammars=["L0", "L0prime", "L1"],
    task_seeds=[0, 1, 2],
    scorers=["sum_logprob"],
)


# ============================================================================
# 5. LADDER ACROSS TASK SAMPLES
# ============================================================================
# Why: the ladder is seed 0 only, so the 4B reversal rests on a single task
# draw, and the 0.125 gap there is five tasks of forty. Five seeds at 1.7B and
# 4B costs little and decides whether the reversal is real or noise. Do this one
# BEFORE writing any more prose about the reversal.

LADDER_GRID = dict(
    models=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
    grammars=["L0", "L0prime", "L1"],
    task_seeds=[0, 1, 2, 3, 4],
    scorers=["sum_logprob"],
)


GRIDS = {
    "scorer": SCORER_GRID,
    "granularity": GRANULARITY_GRID,
    "randseeds": RANDSEED_GRID,
    "family": FAMILY_GRID,
    "ladder": LADDER_GRID,
}


# ============================================================================
# driver
# ============================================================================

def expand(grid):
    keys = ["models", "grammars", "task_seeds", "scorers"]
    cuts = grid.get("cut_seeds", [None])
    for m, g, s, sc, cut in itertools.product(
            grid["models"], grid["grammars"], grid["task_seeds"],
            grid["scorers"], cuts):
        yield dict(model_id=m, grammar_name=g, task_seed=s, scorer=sc,
                   cut_seed=cut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=sorted(GRIDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and exit; use this first")
    ap.add_argument("--smoke", action="store_true",
                    help="run the whole grid with the deterministic dry-run "
                         "scorer: no GPU, no model download. Verifies wiring "
                         "and expressivity certification end to end.")
    args = ap.parse_args()

    grid = GRIDS[args.experiment]
    runs = list(expand(grid))

    if args.experiment == "granularity" and not args.dry_run:
        for name, spec in GRANULARITY_LADDER.items():
            if spec is not None:
                register_grammar(name, spec)

    if args.dry_run:
        print(f"{args.experiment}: {len(runs)} arms")
        for r in runs:
            print("  " + json.dumps(r))
        return

    os.makedirs(args.out, exist_ok=True)
    all_rows, per_task = [], []
    for i, r in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {r}", file=sys.stderr)
        res = run_arm(r["model_id"], r["grammar_name"], r["task_seed"],
                      scorer=r["scorer"], cut_seed=r["cut_seed"],
                      dry_run=args.smoke)
        row = {**r, **{k: v for k, v in res.items() if k != "per_task"}}
        all_rows.append(row)
        for t in res.get("per_task", []):
            per_task.append({**r, **t})

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(all_rows, fh, indent=2)

    if per_task:
        import csv
        cols = sorted({k for row in per_task for k in row})
        with open(os.path.join(args.out, "per_task.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(per_task)
        print("\nper_task.csv is the input to block_inference.py -- make sure the")
        print("`block` column is populated or the block-level test degenerates.")

    print(f"\nwrote {args.out}/summary.json  ({len(all_rows)} arms)")


if __name__ == "__main__":
    main()

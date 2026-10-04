#!/usr/bin/env python3
"""
prepare_records.py -- turn whatever your harness already wrote into the exact
CSV that block_inference.py expects.

You released "the per-task outcome of every arm". This script finds those files,
normalises the column names, and -- the part that matters -- derives the `block`
key that the block-level test needs.

    # look first, commit nothing
    python prepare_records.py results/ --inspect

    # then write the csv
    python prepare_records.py results/ -o per_task.csv

Schema it produces
------------------
    task_id, language, success, block, block_source
    (plus any extra columns it recognised: requires_message, seed, model)

How `block` is derived, in order of preference
----------------------------------------------
1. an explicit `block` / `decision_context` column, if you already log one
2. the FIRST INSTRUCTION the baseline arm emitted, read off the trace.
   This is the right key: Proposition 2 blocks tasks by the argmax at the
   deciding step, and under deterministic decoding that is exactly the first
   emitted instruction. Blocks are defined by the BASELINE arm (L0), because
   that is where the blocking happens, and the same key is then attached to
   every arm for the same task so the pairing stays intact.
3. `requires_message` (the paper's own 32/8 split)
4. `requires_message` inferred from a gold-plan field containing SEND/COMPOSE
5. nothing -- one task per block, which reproduces the numbers you already have
   and defeats the purpose. The script tells you loudly if it lands here.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

# column name aliases seen in the wild
ALIASES = {
    "task_id": ["task_id", "task", "tid", "id", "task_index", "taskid"],
    "language": ["language", "lang", "grammar", "arm", "L", "action_language"],
    "success": ["success", "solved", "correct", "final_state_match", "s", "ok",
                "exact_match", "pass"],
    "block": ["block", "decision_context", "context_key", "block_key"],
    "requires_message": ["requires_message", "needs_message", "message_required",
                         "has_message"],
    "trace": ["trace", "plan", "emitted", "emitted_plan", "instructions",
              "program", "actions", "output_plan"],
    "gold_plan": ["gold_plan", "gold", "reference_plan", "gold_program"],
    "seed": ["seed", "task_seed", "sample_seed"],
    "model": ["model", "model_id", "checkpoint"],
}


def find_key(row, canonical):
    lower = {k.lower(): k for k in row}
    for cand in ALIASES[canonical]:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_any(path):
    """Load one file into a list of dicts, or return []."""
    try:
        if path.endswith(".jsonl"):
            with open(path) as fh:
                return [json.loads(l) for l in fh if l.strip()]
        if path.endswith(".json"):
            with open(path) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            for key in ("records", "per_task", "results", "rows", "tasks"):
                if isinstance(data.get(key), list):
                    return data[key]
            return []
        if path.endswith((".csv", ".tsv")):
            delim = "\t" if path.endswith(".tsv") else ","
            with open(path, newline="") as fh:
                return list(csv.DictReader(fh, delimiter=delim))
    except Exception as exc:                      # noqa: BLE001
        print(f"[skip] {path}: {exc}", file=sys.stderr)
    return []


def gather(root):
    if os.path.isfile(root):
        paths = [root]
    else:
        paths = sorted(
            p for ext in ("json", "jsonl", "csv", "tsv")
            for p in glob.glob(os.path.join(root, "**", f"*.{ext}"),
                               recursive=True)
        )
    out = []
    for p in paths:
        rows = load_any(p)
        rows = [r for r in rows if isinstance(r, dict)]
        if not rows:
            continue
        r0 = rows[0]
        if find_key(r0, "task_id") and find_key(r0, "success"):
            for r in rows:
                r.setdefault("_source_file", os.path.basename(p))
            out.extend(rows)
            print(f"[read] {p}: {len(rows)} rows", file=sys.stderr)
    return out


def to_int(v):
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n", "", "none"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def first_instruction(trace):
    """Pull the first emitted instruction name out of whatever a trace is."""
    if trace is None:
        return None
    if isinstance(trace, str):
        t = trace.strip()
        if not t:
            return None
        if t.startswith("["):
            try:
                trace = json.loads(t)
            except Exception:                     # noqa: BLE001
                trace = [x for x in t.replace(";", "\n").split("\n") if x.strip()]
        else:
            trace = [x for x in t.replace(";", "\n").split("\n") if x.strip()]
    if isinstance(trace, dict):
        trace = trace.get("steps") or trace.get("instructions") or []
    if not isinstance(trace, (list, tuple)) or not trace:
        return None
    head = trace[0]
    if isinstance(head, dict):
        head = head.get("name") or head.get("instruction") or head.get("op") \
            or json.dumps(head, sort_keys=True)
    head = str(head).strip()
    return head.split("(")[0].strip() or None


def normalise(raw, baseline_arm):
    rows, keys = [], (raw[0] if raw else {})
    k_task = find_key(keys, "task_id")
    k_lang = find_key(keys, "language")
    k_succ = find_key(keys, "success")
    k_blk = find_key(keys, "block")
    k_req = find_key(keys, "requires_message")
    k_tr = find_key(keys, "trace")
    k_gold = find_key(keys, "gold_plan")
    k_seed = find_key(keys, "seed")
    k_model = find_key(keys, "model")

    if not k_task or not k_succ:
        sys.exit("could not find task-id / success columns in the records")

    for r in raw:
        row = {
            "task_id": str(r[k_task]).strip(),
            "language": str(r[k_lang]).strip() if k_lang else "unknown",
            "success": to_int(r[k_succ]),
        }
        if k_seed:
            row["seed"] = r[k_seed]
        if k_model:
            row["model"] = r[k_model]
        if k_req:
            row["requires_message"] = to_int(r[k_req])
        elif k_gold:
            g = str(r[k_gold]).upper()
            row["requires_message"] = int("SEND" in g or "COMPOSE" in g
                                          or "NOTIFY" in g)
        row["_block_explicit"] = str(r[k_blk]).strip() if k_blk else ""
        row["_first_instr"] = first_instruction(r[k_tr]) if k_tr else None
        rows.append(row)

    # ---- derive the block key -------------------------------------------
    # key is defined by the BASELINE arm and then attached to every arm.
    src = None
    if any(r["_block_explicit"] for r in rows):
        for r in rows:
            r["block"] = r["_block_explicit"] or "unknown"
        src = "explicit column"
    else:
        base = {r["task_id"]: r["_first_instr"] for r in rows
                if r["language"] == baseline_arm and r["_first_instr"]}
        if len(set(base.values())) > 1:
            for r in rows:
                r["block"] = f"first={base.get(r['task_id'], 'NA')}"
            src = f"first instruction emitted by {baseline_arm}"
        elif any("requires_message" in r for r in rows):
            for r in rows:
                r["block"] = f"msg={r.get('requires_message', 'NA')}"
            src = "requires_message (32/8 split)"
        else:
            for r in rows:
                r["block"] = f"task={r['task_id']}"
            src = "NONE -- one task per block (degenerate)"

    for r in rows:
        r["block_source"] = src
        r.pop("_block_explicit", None)
        r.pop("_first_instr", None)
    return rows, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="results directory or a single records file")
    ap.add_argument("-o", "--out", default="per_task.csv")
    ap.add_argument("--baseline-arm", default="L0",
                    help="arm whose failure structure defines the blocks")
    ap.add_argument("--inspect", action="store_true",
                    help="report what was found and exit without writing")
    args = ap.parse_args()

    raw = gather(args.root)
    if not raw:
        sys.exit(f"no per-task records found under {args.root!r}")

    rows, src = normalise(raw, args.baseline_arm)

    langs = Counter(r["language"] for r in rows)
    blocks = defaultdict(set)
    for r in rows:
        blocks[r["block"]].add(r["task_id"])

    print(f"\nrows            {len(rows)}")
    print(f"arms            {dict(langs)}")
    print(f"tasks           {len({r['task_id'] for r in rows})}")
    print(f"block source    {src}")
    print(f"blocks          {len(blocks)}  "
          f"sizes {sorted((len(v) for v in blocks.values()), reverse=True)[:8]}")

    if src.startswith("NONE"):
        print("\n!! No usable block key. The block-level test will just")
        print("!! reproduce the numbers already in the paper. Fix by logging")
        print("!! the first emitted instruction per task in the L0 arm, or a")
        print("!! requires_message flag, then re-run this.")
    elif len(blocks) > len(rows) / 4:
        print("\n[note] many small blocks -- check that the key is the argmax at")
        print("[note] the deciding step, not something task-specific.")

    if args.inspect:
        return

    cols = sorted({k for r in rows for k in r})
    order = [c for c in ("task_id", "language", "success", "block",
                         "block_source", "requires_message", "seed", "model")
             if c in cols] + [c for c in cols if c not in
                              ("task_id", "language", "success", "block",
                               "block_source", "requires_message", "seed",
                               "model")]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}\n")
    print("next:")
    print(f"  python block_inference.py {args.out} --arm-a L0 --arm-b L1")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_extra.py -- the two runs that most change the paper's reception.

  (1) SECOND MODEL FAMILY   L0 / L0' / L1 on a non-Qwen family, seeds 0-4
  (2) LADDER SEEDS          Qwen3-1.7B and Qwen3-4B on seeds 1-4 (seed 0 exists)

Everything below the ADAPTER block is complete and follows Algorithm 1 of the
paper exactly: enumerated legal candidates, summed log-probability of the
candidate's own tokens given the prompt, argmax, STOP scored as a structural
decision, Htot accumulated in nats, halt on illegal transition, step budget 8,
success = exact final-state equality.

The ADAPTER block (~10 lines) is the only thing that needs to know your code.
Wire it, then:

    python run_extra.py --selftest                    # plumbing check, no GPU, no models
    python run_extra.py --plan family                 # prints what will run
    python run_extra.py --plan family --go            # runs it
    python run_extra.py --plan ladder --go
    python run_extra.py --models HuggingFaceTB/SmolLM2-360M-Instruct --seeds 0 --languages L1 --go

Output: results/extra/<model_slug>/seed<k>/<lang>.jsonl, one record per task,
resumable (existing records are skipped). Feed the directory to
extra_analysis.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

# =============================================================================
# ADAPTER -- wire these to the released bundle. Nothing else needs editing.
# =============================================================================
#
# Your code already contains (Appendix "Reproducibility"): the task generator
# with the 70 task definitions, the five grammars and every constrained
# variant, the executor, and the prompt templates. Expose them through the two
# protocols below. If your decode loop is already a function, you can instead
# set RUN_TASK to it and skip the protocols entirely (see the bottom of this
# block).


class Executor(Protocol):
    def initial_state(self, task: Any) -> Any: ...
    def step(self, state: Any, primitive_call: Any) -> Any:
        """Return the new state, or None if the transition is illegal."""
        ...
    def is_gold(self, state: Any, task: Any) -> bool:
        """Exact final-state agreement on every record and field."""
        ...


class Language(Protocol):
    name: str
    def legal_instructions(self, history: list) -> list[str]:
        """Instruction names legal after `history` (L0': successor rule).
        Do NOT include STOP; the loop adds it."""
        ...
    def slots(self, instr: str) -> list[tuple[str, list[str]]]:
        """[(slot_name, [domain values as scored strings]), ...] in signature order."""
        ...
    def prompt(self, task: Any, history: list,
               partial: tuple[str, list[str]] | None = None) -> str:
        """The exact prompt string of this arm: signatures, address book,
        already-resolved set, task utterance, plan so far; `partial` is
        (instr, values_filled_so_far) when scoring a slot."""
        ...
    def candidate_text(self, instr: str, value: str | None = None) -> str:
        """The string that is actually scored for a candidate: e.g. 'SEND(' for
        a structural candidate or 'email' for a slot value. Must match what the
        released loop scores, token for token."""
        ...
    def expand(self, instr: str, values: list[str]) -> list[Any]:
        """Deterministic compilation to primitive calls (identity for L0)."""
        ...


def load_tasks(seed: int) -> list[dict]:
    """Return the 70 tasks for `seed` as dicts with at least:
       id, split ('seen'|'cf'), cf_dim (None for seen), source_id (for cf),
       task (the object your executor/language understand).
    Seen tasks are the first 40 enumeration points under `seed`."""
    raise NotImplementedError(
        "ADAPTER: return the task list from your generator, e.g.\n"
        "    from actionisa.tasks import generate\n"
        "    return generate(seed=seed)   # 40 seen + 30 counterfactual")


def load_language(name: str) -> Language:
    """name in {'L0', 'L0p', 'L1', 'L2', 'L3', 'Lrand'}; 'L0p' is L0'."""
    raise NotImplementedError(
        "ADAPTER: return your Language object, e.g.\n"
        "    from actionisa.grammars import LANGUAGES\n"
        "    return LANGUAGES[name]")


def load_executor() -> Executor:
    raise NotImplementedError(
        "ADAPTER: return your executor, e.g.\n"
        "    from actionisa.sandbox import Executor\n"
        "    return Executor()")


# If your decode loop is already a function that returns one per-task record,
# point RUN_TASK at it and the loop below is bypassed. Signature:
#   RUN_TASK(model, tokenizer, language_name: str, task: dict) -> dict with keys
#   success (bool), htot (float), kstruct (int), kparam (int), n_instr (int),
#   halted_illegal (bool), history (list)
RUN_TASK = None

# =============================================================================
# End of ADAPTER.
# =============================================================================

STEP_BUDGET = 8
STOP = "STOP"

# Model plans. Revisions for the Qwen ladder are the ones in Table "Exact
# experimental configuration"; pin them so the seed-0 rows are reproduced.
QWEN_LADDER = [
    ("Qwen/Qwen3-1.7B", "70d244cc"),
    ("Qwen/Qwen3-4B", "1cfa9a72"),
]
# Second family: open, ungated, sub-billion to ~1.7B, one family so scale is
# isolated within it (same reason the paper used the Qwen ladder). Gated
# alternatives (need HF_TOKEN + licence): google/gemma-3-270m, google/gemma-3-1b-it,
# meta-llama/Llama-3.2-1B-Instruct.
SECOND_FAMILY = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct", None),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", None),
]
PLANS = {
    "family": dict(models=SECOND_FAMILY, seeds=[0, 1, 2, 3, 4], languages=["L0", "L0p", "L1"]),
    "ladder": dict(models=QWEN_LADDER, seeds=[1, 2, 3, 4], languages=["L0", "L0p", "L1"]),
}


# ----------------------------------------------------------------------------- scoring
def _entropy_from_logscores(scores: Sequence[float]) -> float:
    m = max(scores)
    z = [math.exp(s - m) for s in scores]
    Z = sum(z)
    p = [v / Z for v in z]
    return -sum(pi * math.log(pi) for pi in p if pi > 0.0)


class HFScorer:
    """Score(M, prompt, C): summed log p(c_i | prompt, c_<i) for every c in C,
    one batched forward pass. Deterministic; no sampling anywhere."""

    def __init__(self, model_id: str, revision: str | None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if self.device == "cuda" else torch.float32)
        kw = dict(revision=revision) if revision else {}
        self.tok = AutoTokenizer.from_pretrained(model_id, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kw).to(self.device).eval()
        self.model_id, self.revision = model_id, revision
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.ctx = getattr(self.model.config, "max_position_embeddings", 4096)

    @property
    def resolved_revision(self) -> str:
        cfg = getattr(self.model, "config", None)
        return self.revision or getattr(cfg, "_commit_hash", None) or "unpinned"

    def score(self, prompt: str, candidates: Sequence[str]) -> list[float]:
        torch = self.torch
        p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        seqs, cand_lens = [], []
        for c in candidates:
            c_ids = self.tok(c, add_special_tokens=False)["input_ids"]
            if len(c_ids) == 0:
                raise ValueError(f"candidate tokenises to nothing: {c!r}")
            seqs.append(p_ids + c_ids)
            cand_lens.append(len(c_ids))
        L = max(len(s) for s in seqs)
        if L > self.ctx:
            raise ValueError(f"prompt+candidate {L} tokens exceeds context {self.ctx}")
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):               # right padding
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        out = []
        for i, s in enumerate(seqs):
            n = cand_lens[i]
            start = len(s) - n                      # first candidate token position
            tgt = ids[i, start:len(s)]
            # token at position t is predicted from logits at t-1
            lp = logp[i, start - 1:len(s) - 1, :].gather(1, tgt.unsqueeze(1)).squeeze(1)
            out.append(float(lp.sum().item()))
        return out


# ----------------------------------------------------------------------------- the loop
def run_task_algorithm1(scorer, lang: Language, ex: Executor, task: dict) -> dict:
    """Algorithm 1, verbatim. Returns the per-task record."""
    t = task["task"]
    s = ex.initial_state(t)
    history: list = []
    htot, kstruct, kparam = 0.0, 0, 0
    halted_illegal = False
    decisions = []                                   # (kind, |A|, H) for block-level analysis
    for k in range(STEP_BUDGET):
        A = list(lang.legal_instructions(history)) + [STOP]
        cand = [lang.candidate_text(a) for a in A]
        ell = scorer.score(lang.prompt(t, history), cand)
        H = _entropy_from_logscores(ell)
        htot += H; kstruct += 1
        decisions.append(("struct", len(A), H))
        x = A[max(range(len(A)), key=lambda i: ell[i])]
        if x == STOP:
            break
        values: list[str] = []
        for slot_name, domain in lang.slots(x):
            cand = [lang.candidate_text(x, v) for v in domain]
            ell = scorer.score(lang.prompt(t, history, partial=(x, values)), cand)
            H = _entropy_from_logscores(ell)
            htot += H; kparam += 1
            decisions.append(("param", len(domain), H))
            values.append(domain[max(range(len(domain)), key=lambda i: ell[i])])
        history.append((x, values))
        s_next = s
        for prim in lang.expand(x, values):
            s_next = ex.step(s_next, prim)
            if s_next is None:
                break
        if s_next is None:
            halted_illegal = True
            break
        s = s_next
    return dict(
        success=bool(not halted_illegal and ex.is_gold(s, t)),
        htot=round(htot, 6), kstruct=kstruct, kparam=kparam,
        n_instr=len(history), halted_illegal=halted_illegal,
        history=[[x, v] for x, v in history], decisions=decisions,
    )


# ----------------------------------------------------------------------------- driver
def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def out_path(root: Path, model_id: str, seed: int, lang: str) -> Path:
    return root / slug(model_id) / f"seed{seed}" / f"{lang}.jsonl"


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["task_id"] for l in path.read_text().splitlines() if l.strip()}


def run_plan(models, seeds, languages, root: Path, dry: bool):
    for model_id, rev in models:
        todo = [(sd, lg) for sd in seeds for lg in languages]
        print(f"\n== {model_id} @ {rev or 'latest'}: {len(todo)} (seed, language) cells", flush=True)
        if dry:
            for sd, lg in todo:
                p = out_path(root, model_id, sd, lg)
                print(f"   seed {sd} {lg:5s} -> {p}  [{len(existing_ids(p))} done]")
            continue
        scorer = HFScorer(model_id, rev)
        print(f"   loaded on {scorer.device}, revision {scorer.resolved_revision}", flush=True)
        ex = load_executor()
        for sd in seeds:
            tasks = load_tasks(sd)
            assert len(tasks) == 70, f"expected 70 tasks (40 seen + 30 cf), got {len(tasks)}"
            for lg in languages:
                lang = load_language(lg)
                p = out_path(root, model_id, sd, lg)
                p.parent.mkdir(parents=True, exist_ok=True)
                done = existing_ids(p)
                t0 = time.time()
                with p.open("a") as f:
                    for task in tasks:
                        if task["id"] in done:
                            continue
                        rec = RUN_TASK(scorer.model, scorer.tok, lg, task) if RUN_TASK else \
                            run_task_algorithm1(scorer, lang, ex, task)
                        rec.update(model=model_id, revision=scorer.resolved_revision, seed=sd,
                                   language=lg, task_id=task["id"], split=task["split"],
                                   cf_dim=task.get("cf_dim"), source_id=task.get("source_id"))
                        f.write(json.dumps(rec) + "\n"); f.flush()
                n = len(existing_ids(p))
                S_seen = _quick_S(p, "seen")
                print(f"   seed {sd} {lg:5s} {n:2d}/70  S_seen={S_seen:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        del scorer
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def _quick_S(path: Path, split: str) -> float:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["split"] == split]
    return sum(r["success"] for r in recs) / max(1, len(recs))


# ----------------------------------------------------------------------------- selftest
def _selftest():
    """Exercises the loop + record schema with a toy language, executor and
    scorer. It does NOT use your sandbox; it only proves the plumbing runs."""
    import random
    rng = random.Random(0)

    class ToyEx:
        def initial_state(self, t): return []
        def step(self, s, prim):
            if prim == "SEND" and "COMPOSE" not in s:
                return None                              # the precondition L0 trips on
            return s + [prim]
        def is_gold(self, s, t): return s == t["gold"]

    class ToyLang:
        def __init__(self, name, bundled):
            self.name, self.bundled = name, bundled
        def legal_instructions(self, h):
            return ["DO_ALL"] if self.bundled else ["RESOLVE", "CREATE", "COMPOSE", "SEND"]
        def slots(self, x):
            return [("order", ["event_first", "message_first"])] if self.bundled else [("who", ["sara", "ali"])]
        def prompt(self, t, h, partial=None):
            return f"task: {t['utt']} plan: {h} partial: {partial}"
        def candidate_text(self, x, v=None): return x if v is None else v
        def expand(self, x, values):
            if self.bundled:
                return ["RESOLVE", "CREATE", "COMPOSE", "SEND"]
            return [x]

    class ToyScorer:
        device = "cpu"; resolved_revision = "toy"; model = None; tok = None
        def score(self, prompt, cands): return [rng.gauss(0, 1) for _ in cands]

    tasks = [dict(id=f"t{i}", split="seen" if i < 40 else "cf", cf_dim=None if i < 40 else "ordering",
                  source_id=None if i < 40 else f"t{i-40}", task=dict(utt=f"u{i}", gold=["RESOLVE", "CREATE", "COMPOSE", "SEND"]))
             for i in range(70)]
    ex, sc = ToyEx(), ToyScorer()
    for name, bundled in [("L0", False), ("L1", True)]:
        recs = [run_task_algorithm1(sc, ToyLang(name, bundled), ex, t) for t in tasks]
        S = sum(r["success"] for r in recs) / len(recs)
        assert all(set(r) >= {"success", "htot", "kstruct", "kparam", "n_instr", "halted_illegal", "history"} for r in recs)
        assert all(r["kstruct"] <= STEP_BUDGET for r in recs)
        print(f"selftest {name}: S={S:.3f}  mean Htot={sum(r['htot'] for r in recs)/len(recs):.3f}  "
              f"mean Kstruct={sum(r['kstruct'] for r in recs)/len(recs):.2f}")
    print("selftest OK -- loop, entropy accounting and record schema run end to end.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=list(PLANS), help="family | ladder")
    ap.add_argument("--models", nargs="*", help="override: id or id@revision")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--languages", nargs="*", help="subset of L0 L0p L1 (L2 L3 Lrand also accepted)")
    ap.add_argument("--out", default="results/extra")
    ap.add_argument("--go", action="store_true", help="actually run (default is a dry listing)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    if not a.plan and not a.models:
        ap.error("give --plan family|ladder or --models ...")
    plan = dict(PLANS[a.plan]) if a.plan else dict(models=[], seeds=[0], languages=["L0", "L0p", "L1"])
    if a.models:
        plan["models"] = [tuple(m.split("@")) if "@" in m else (m, None) for m in a.models]
    if a.seeds is not None:
        plan["seeds"] = a.seeds
    if a.languages:
        plan["languages"] = a.languages
    root = Path(a.out)
    run_plan(plan["models"], plan["seeds"], plan["languages"], root, dry=not a.go)
    if not a.go:
        print("\n(dry listing; add --go to run)")


if __name__ == "__main__":
    main()

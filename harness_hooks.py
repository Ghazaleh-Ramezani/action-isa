#!/usr/bin/env python3
"""
harness_hooks.py -- wires ablations.py to the released harness.

ablations.py ships three HOOK functions raising NotImplementedError. This module
implements all three against r3_all_in_one.py (which itself imports the r0
harness), and ablations.py imports them automatically. Nothing here changes the
decision loop: run_arm calls the same run_task_r3 the paper's numbers came from.

Requires in the same directory:
    r0_experiment.py      task builder, L0-L3, L_rand, oracle, HFScorer
    r3_all_in_one.py      successor-constrained grammars, run_task_r3

Grammar names understood by run_arm:
    L0, L1, L2, L3            from r0_experiment.all_languages
    L0prime                   successor-constrained (pre+stop+order)
    Lrand_matched             random-boundary control, cut_seed selects the cut
    G1, G2, G3                registered at runtime by ablations.py
"""

from __future__ import annotations

import os
import sys

# Make this file's own directory importable. Needed when python runs in
# isolated mode (-I / -P), where the script directory is not added to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import r0_experiment as H
    import r3_all_in_one as R3
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"{exc.name}.py not found next to harness_hooks.py.\n"
        f"Put r0_experiment.py and r3_all_in_one.py in {_HERE} "
        f"(both ship in the reproduction bundle)."
    ) from exc

# --------------------------------------------------------------------------
# grammar registry
# --------------------------------------------------------------------------

_REGISTERED: dict = {}


def _base_languages(cut_seed: int = 0):
    return {l.name: l for l in H.all_languages(cut_seed)}


def _l0prime(cut_seed: int = 0):
    """Successor-constrained control: L0's instructions, restricted successors.

    All three constraint families on (preconditions, stopping, ordering), which
    is the L0' of the paper. Nothing is bundled, so the structural form of every
    choice survives while |A_k| shrinks at each one.
    """
    return R3.make_lang("L0prime",
                        lambda prev: R3.successors(prev, pre=True, stop=True,
                                                   order=True),
                        seed=cut_seed)


def load_grammar(name, cut_seed: int = 0):
    """HOOK 1: return the Language object for a registered name."""
    if name in _REGISTERED:
        return _REGISTERED[name]

    if name == "L0prime":
        return _l0prime(cut_seed)

    if name in ("Lrand_matched", "L_rand", "Lrand"):
        # cut_seed selects which random boundary draw; this is the whole point
        # of the randseeds experiment.
        return H.make_Lrand(seed=cut_seed if cut_seed is not None else 0)

    base = _base_languages(cut_seed if cut_seed is not None else 0)
    if name in base:
        return base[name]
    if name == "G0":
        return base["L0"]
    if name == "G4":
        return base["L1"]

    raise KeyError(
        f"unknown grammar {name!r}; known: {sorted(set(base) | set(_REGISTERED) | {'L0prime', 'Lrand_matched', 'G0', 'G4'})}"
    )


def register_grammar(name, spec):
    """HOOK 2: build a Language from a granularity-ladder spec and register it.

    spec = {"instructions": [{"name": str, "slots": {slot: [domain]}}, ...]}

    Each bundled instruction expands to the concatenation of the primitives its
    name names, in dependency order -- exactly how L1 is built in the harness.
    Slot order follows the spec dict order, which is the signature order the
    decoding loop consumes.
    """
    # Map a bundle name to the primitive sequence it expands to.
    #
    # NOTIFY is conditional. Welding GET_LINK unconditionally into the message
    # makes every include_link=False task unreachable (E_sem = 0.40 on seed 0,
    # 24 of 40 tasks), because the language would have no way to send a message
    # without first fetching a link. The recipe requires E_syn = E_sem = 1 on
    # every rung, so the link fetch is gated on the bundle's own include_link
    # slot: the rung still demotes ordering to a slot, but it does not silently
    # remove an option the executor supports.
    PARTS = {
        "RESOLVE":            ["RESOLVE"],
        "CREATE_EVENT":       ["CREATE_EVENT"],
        "GET_LINK":           ["GET_LINK"],
        "COMPOSE":            ["COMPOSE"],
        "SEND":               ["SEND"],
        "RESOLVE_AND_CREATE": ["RESOLVE", "CREATE_EVENT"],
        "COMPOSE_AND_SEND":   ["COMPOSE", "SEND"],
        "NOTIFY":             ["GET_LINK", "COMPOSE", "SEND"],
    }
    CONDITIONAL = {
        # bundle -> {primitive: slot that must be truthy for it to be emitted}
        "NOTIFY": {"GET_LINK": "compose_include_link"},
    }

    instructions = []
    for entry in spec["instructions"]:
        iname = entry["name"]
        if iname not in PARTS:
            raise KeyError(f"no expansion known for bundle {iname!r}; "
                           f"add it to PARTS in harness_hooks.register_grammar")
        chunk = PARTS[iname]

        # Domains ALWAYS come from the executor, never from the spec. The spec's
        # CONTACTS/TIMES/TOPICS in ablations.py are placeholder tokens ("t1",
        # "m1"); building a grammar over those yields E_sem = 0, because no
        # sequence can reach the gold state. The spec declares which primitives
        # a bundle contains; the executor declares what values its slots accept.
        slots = []
        for pn in chunk:
            for sname, vals in H.PRIMITIVES[pn]["slots"]:
                slots.append((f"{pn.lower()}_{sname}", list(vals)))

        slot_names = [s for s, _ in slots]
        gates = CONDITIONAL.get(iname, {})

        def _make_expand(chunk_, slot_names_, gates_):
            def _e(args, spec_):
                # resolve by slot NAME, not position, so a conditionally
                # omitted primitive cannot shift the remaining arguments
                val = dict(zip(slot_names_, args))
                out = []
                for pn in chunk_:
                    gate = gates_.get(pn)
                    if gate is not None and not val.get(gate, False):
                        continue
                    out.append((pn, [val[f"{pn.lower()}_{sname}"]
                                     for sname, _ in H.PRIMITIVES[pn]["slots"]]))
                return out
            return _e

        instructions.append(H.Instruction(
            name=iname, slots=slots,
            expand=_make_expand(chunk, slot_names, gates)))

    lang = H.Language(name=name, instructions=instructions,
                      expressible_dims=frozenset(H.CF_DIMS),
                      description=f"granularity rung {name}")
    _REGISTERED[name] = lang
    return lang


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------

class MeanLogProbScorer(H.HFScorer):
    """Length-normalised variant: score(a) = (1/|a|) * sum_i log p(a_i | ...).

    Subclasses the released scorer so tokenisation, batching and device
    handling are identical; only the reduction over candidate tokens changes.
    This is the scorer ablation's treatment arm.
    """

    def score(self, prompt, candidates):
        torch = self.torch
        p_ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        seqs, starts = [], []
        for c in candidates:
            c_ids = self.tok(self._norm(c), add_special_tokens=False,
                             return_tensors="pt").input_ids[0]
            seqs.append(torch.cat([p_ids, c_ids]))
            starts.append(len(p_ids))
        maxlen = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            batch[i, : len(s)] = s
            mask[i, : len(s)] = 1
        batch, mask = batch.to(self.device), mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=batch, attention_mask=mask).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for i, s in enumerate(seqs):
            tot, n = 0.0, 0
            for pos in range(starts[i], len(s)):
                tot += logprobs[i, pos - 1, batch[i, pos]].item()
                n += 1
            out.append(tot / max(1, n))          # <- the only change
        return out


_SCORER_CACHE: dict = {}


def _get_scorer(model_id, scorer_name, dry_run=False):
    key = (model_id, scorer_name, dry_run)
    if key in _SCORER_CACHE:
        return _SCORER_CACHE[key]
    if dry_run:
        obj = H.DryRunScorer(seed=0)
    elif scorer_name == "mean_logprob":
        obj = MeanLogProbScorer(model_id)
    elif scorer_name == "sum_logprob":
        obj = H.HFScorer(model_id)
    else:
        raise ValueError(f"unknown scorer {scorer_name!r}")
    _SCORER_CACHE[key] = obj
    return obj


# --------------------------------------------------------------------------
# run one arm
# --------------------------------------------------------------------------

def run_arm(model_id, grammar_name, task_seed, *, scorer="sum_logprob",
            task_set="seen", cut_seed=None, dry_run=False,
            certify_expressivity=True):
    """HOOK 3: run one arm and return the record dict ablations.py expects.

    Returns S, S_cf, H_tot, K_struct, K_param, E_syn, E_sem plus a per_task list
    carrying the `block` column block_inference.py needs.
    """
    seen, cf = H.build_tasks(seed=task_seed)
    lang = load_grammar(grammar_name, cut_seed if cut_seed is not None else 0)
    sc = _get_scorer(model_id, scorer, dry_run=dry_run)

    # expressivity certification: the recipe requires E_syn = E_sem = 1 on every
    # rung, and a bundling bug shows up here rather than in the model numbers.
    e_syn = sum(H.e_syntax(lang, s) for s in seen) / len(seen)
    e_sem = (sum(H.e_semantic(lang, s) for s in seen) / len(seen)
             if certify_expressivity else None)

    recs = [R3.run_task_r3(sc, lang, spec) for spec in seen]
    recs_cf = [R3.run_task_r3(sc, lang, spec) for spec in cf]
    n = len(recs)

    per_task = []
    for spec, r in zip(seen, recs):
        # Column names match what block_inference.py reads: `language`,
        # `task_id`, `success`, plus a `block` column.
        #
        # task_id is qualified by seed and model. The harness names tasks
        # seen000..seen039 per seed, so an unqualified id would silently pair
        # seed 0's seen000 with seed 1's seen000 -- two different tasks.
        #
        # `block` is the argmax-invariant prefix at the deciding step: whether
        # the task needs a message (the 32/8 split the paper describes) plus the
        # first instruction chosen. Tasks sharing it share a decision context,
        # which is the unit block_inference.py permutes over.
        first = r.trace[0].split("(")[0] if r.trace else "NONE"
        needs_msg = int(spec.channel is not None)
        per_task.append({
            "language": grammar_name,
            "task_id": f"{model_id}|s{task_seed}|{r.tid}",
            "success": int(r.success),
            "block": f"msg{needs_msg}|{first}",
            "requires_message": needs_msg,
            "k_struct": r.k_struct,
            "k_param": r.k_param,
            "h_tot": round(r.h_tot, 6),
        })

    return {
        "S": sum(r.success for r in recs) / n,
        "S_cf": sum(r.success for r in recs_cf) / max(1, len(recs_cf)),
        "H_tot": sum(r.h_tot for r in recs) / n,
        "K_struct": sum(r.k_struct for r in recs) / n,
        "K_param": sum(r.k_param for r in recs) / n,
        "E_syn": e_syn,
        "E_sem": e_sem,
        "prompt_tokens": sum(r.prompt_tokens for r in recs) / n,
        "per_task": per_task,
    }


if __name__ == "__main__":
    # smoke test with the deterministic dry-run scorer: no GPU, no model
    # download. Verifies the hooks are wired and expressivity certifies.
    print("smoke test: dry-run scorer, seed 0, 3 grammars")
    for g in ["L0", "L0prime", "L1"]:
        out = run_arm("dry", g, 0, dry_run=True)
        print(f"  {g:9s} S={out['S']:.3f}  E_syn={out['E_syn']:.2f} "
              f"E_sem={out['E_sem']:.2f}  K_struct={out['K_struct']:.2f}  "
              f"blocks={len({t['block'] for t in out['per_task']})}")
    print("hooks OK")

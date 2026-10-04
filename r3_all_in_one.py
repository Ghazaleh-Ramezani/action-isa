#!/usr/bin/env python3
"""

R3 confirmatory run -- SINGLE FILE version.

Everything is here: sandbox, action languages, executor, and experiment runner.
Nothing else to upload.

    pip install torch transformers accelerate
    python r3_all_in_one.py --parts D E

Needs an NVIDIA GPU (a T4 is fine). Checkpoints into r3_ckpt/ after every arm,
so re-running the same command resumes rather than restarting.

Produces: r3_results.json -- that one file is what to send back.
"""

from __future__ import annotations


import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)) or "."

# =====================================================================
# PART 1 of 2 -- sandbox, action languages, executor
# =====================================================================

"""
R0: Structural vs parametric action representations for small tool-use models.

Self-contained. No external data required.

Usage:
    python r0_experiment.py --dry-run          # no GPU, validates pipeline
    python r0_experiment.py --model Qwen/Qwen3-0.6B

Design notes that matter for interpreting the output:
  * The entity universe is CLOSED (fixed contacts/times/topics). This makes every
    decision enumerable, so p_k over the legal action set is exact rather than
    approximated. It also makes the language-understanding part of the task
    trivial, which is deliberate: we are isolating decision FORM, not NLU.
    Report this as a limitation.
  * Decisions are made by scoring each legal candidate under the model and taking
    argmax. This is constrained decoding, applied identically to ALL languages
    including L0.
  * K_struct counts instruction-selection decisions (including STOP).
    K_param counts slot-filling decisions.
"""


import argparse
import itertools
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

# =====================================================================
# 1. Environment
# =====================================================================

CONTACTS = ["Sara", "Ali", "Maryam", "Reza"]
TIMES = ["Monday 10am", "Tuesday 2pm", "Wednesday 4pm", "Friday 9am"]
TOPICS = ["budget review", "design sync", "hiring", "roadmap"]
CHANNELS = ["email", "slack"]
ORDERS = ["event_first", "message_first"]


@dataclass(frozen=True)
class State:
    """Immutable, hashable device state."""
    resolved: FrozenSet[str] = frozenset()          # contact names resolved to ids
    events: FrozenSet[Tuple[str, str, str]] = frozenset()   # (topic, time, attendee)
    links: FrozenSet[Tuple[str, str, str]] = frozenset()    # events whose link was fetched
    drafts: FrozenSet[Tuple[str, str, bool, str]] = frozenset()  # (recipient, topic, has_link, channel)
    sent: FrozenSet[Tuple[str, str, bool, str]] = frozenset()

    def key(self) -> str:
        return json.dumps(
            {
                "resolved": sorted(self.resolved),
                "events": sorted(map(list, self.events)),
                "links": sorted(map(list, self.links)),
                "drafts": sorted([[a, b, c, d] for a, b, c, d in self.drafts]),
                "sent": sorted([[a, b, c, d] for a, b, c, d in self.sent]),
            },
            sort_keys=True,
        )


# ---- primitives ------------------------------------------------------
# Each returns a new State, or None if the precondition fails.

def p_resolve(s: State, name: str) -> Optional[State]:
    if name not in CONTACTS:
        return None
    return replace(s, resolved=s.resolved | {name})


def p_create_event(s: State, topic: str, time: str, attendee: str) -> Optional[State]:
    if attendee not in s.resolved:
        return None                      # must resolve the contact first
    ev = (topic, time, attendee)
    if ev in s.events:
        return None                      # no duplicate events
    return replace(s, events=s.events | {ev})


def p_get_link(s: State, topic: str, time: str, attendee: str) -> Optional[State]:
    ev = (topic, time, attendee)
    if ev not in s.events:
        return None
    if ev in s.links:
        return None
    return replace(s, links=s.links | {ev})


def p_compose(s: State, recipient: str, topic: str, include_link: bool,
              channel: str) -> Optional[State]:
    if channel not in CHANNELS:
        return None
    if include_link:
        # data dependency: the link must have been fetched for some event on this topic
        if not any(t == topic for (t, _, _) in s.links):
            return None
    d = (recipient, topic, include_link, channel)
    if d in s.drafts:
        return None
    return replace(s, drafts=s.drafts | {d})


def p_send(s: State, recipient: str, topic: str, include_link: bool,
           channel: str) -> Optional[State]:
    d = (recipient, topic, include_link, channel)
    if d not in s.drafts:
        return None
    if d in s.sent:
        return None
    return replace(s, sent=s.sent | {d})


PRIMITIVES: Dict[str, Dict[str, Any]] = {
    "RESOLVE":      {"fn": p_resolve,      "slots": [("name", CONTACTS)]},
    "CREATE_EVENT": {"fn": p_create_event, "slots": [("topic", TOPICS), ("time", TIMES),
                                                     ("attendee", CONTACTS)]},
    "GET_LINK":     {"fn": p_get_link,     "slots": [("topic", TOPICS), ("time", TIMES),
                                                     ("attendee", CONTACTS)]},
    "COMPOSE":      {"fn": p_compose,      "slots": [("recipient", CONTACTS), ("topic", TOPICS),
                                                     ("include_link", [True, False]),
                                                     ("channel", CHANNELS)]},
    "SEND":         {"fn": p_send,         "slots": [("recipient", CONTACTS), ("topic", TOPICS),
                                                     ("include_link", [True, False]),
                                                     ("channel", CHANNELS)]},
}


def apply_primitive(s: State, name: str, args: Sequence[Any]) -> Optional[State]:
    spec = PRIMITIVES.get(name)
    if spec is None:
        return None
    if len(args) != len(spec["slots"]):
        return None
    return spec["fn"](s, *args)


def run_program(s0: State, program: Sequence[Tuple[str, Sequence[Any]]]) -> Optional[State]:
    s = s0
    for name, args in program:
        s = apply_primitive(s, name, args)
        if s is None:
            return None
    return s


# =====================================================================
# 2. Tasks
# =====================================================================

@dataclass(frozen=True)
class TaskSpec:
    contact: str
    time: str
    topic: str
    channel: Optional[str]        # None means "do not notify"
    include_link: bool
    order: str                    # event_first | message_first
    pre_resolved: bool            # contact already in the address book cache
    tid: str = ""
    dim: str = "seen"             # which counterfactual dimension, or "seen"
    parent: str = ""


def initial_state(spec: TaskSpec) -> State:
    return State(resolved=frozenset({spec.contact}) if spec.pre_resolved else frozenset())


def gold_program(spec: TaskSpec) -> List[Tuple[str, Sequence[Any]]]:
    prog: List[Tuple[str, Sequence[Any]]] = []
    if not spec.pre_resolved:
        prog.append(("RESOLVE", [spec.contact]))

    event_block = [("CREATE_EVENT", [spec.topic, spec.time, spec.contact])]
    if spec.include_link:
        event_block.append(("GET_LINK", [spec.topic, spec.time, spec.contact]))

    msg_block: List[Tuple[str, Sequence[Any]]] = []
    if spec.channel is not None:
        msg_block = [
            ("COMPOSE", [spec.contact, spec.topic, spec.include_link, spec.channel]),
            ("SEND",    [spec.contact, spec.topic, spec.include_link, spec.channel]),
        ]

    if spec.order == "message_first" and msg_block and not spec.include_link:
        prog += msg_block + event_block
    else:
        prog += event_block + msg_block
    return prog


def gold_state(spec: TaskSpec) -> Optional[State]:
    return run_program(initial_state(spec), gold_program(spec))


def utterance(spec: TaskSpec) -> str:
    """Templated natural language. Deliberately simple; see module docstring."""
    base = f"Set up a {spec.topic} meeting with {spec.contact} on {spec.time}"
    if spec.channel is None:
        return base + ", and don't notify anyone."
    link = " including the event link" if spec.include_link else " without the event link"
    order = ""
    if spec.order == "message_first" and not spec.include_link:
        order = " Send the message before creating the event."
    return f"{base}, then send them a {spec.channel} message{link}.{order}"


# ---- counterfactual mutations ---------------------------------------

CF_DIMS = ["channel", "side_effect", "ordering", "primitive_sub", "data_dependency"]


def mutate(spec: TaskSpec, dim: str) -> Optional[TaskSpec]:
    if dim == "channel":
        if spec.channel is None:
            return None
        other = "slack" if spec.channel == "email" else "email"
        return replace(spec, channel=other, dim=dim, parent=spec.tid)
    if dim == "side_effect":
        if spec.channel is None:
            return None
        return replace(spec, channel=None, include_link=False, dim=dim, parent=spec.tid)
    if dim == "ordering":
        if spec.channel is None or spec.include_link:
            return None
        new_order = "message_first" if spec.order == "event_first" else "event_first"
        return replace(spec, order=new_order, dim=dim, parent=spec.tid)
    if dim == "primitive_sub":
        return replace(spec, pre_resolved=not spec.pre_resolved, dim=dim, parent=spec.tid)
    if dim == "data_dependency":
        if spec.channel is None:
            return None
        return replace(spec, include_link=not spec.include_link,
                       order="event_first", dim=dim, parent=spec.tid)
    return None


def build_tasks(n_seen: int = 40, n_cf_per_dim: int = 6,
                seed: int = 0) -> Tuple[List[TaskSpec], List[TaskSpec]]:
    rng = random.Random(seed)
    pool: List[TaskSpec] = []
    for c, t, tp in itertools.product(CONTACTS, TIMES, TOPICS):
        for ch in CHANNELS + [None]:
            for link in [True, False]:
                for order in ORDERS:
                    for pre in [True, False]:
                        if ch is None and link:
                            continue
                        if order == "message_first" and (ch is None or link):
                            continue
                        pool.append(TaskSpec(c, t, tp, ch, link, order, pre))
    rng.shuffle(pool)

    seen: List[TaskSpec] = []
    for i, sp in enumerate(pool):
        if len(seen) >= n_seen:
            break
        sp = replace(sp, tid=f"seen{len(seen):03d}")
        if gold_state(sp) is None:
            continue
        seen.append(sp)

    cf: List[TaskSpec] = []
    for dim in CF_DIMS:
        made = 0
        for sp in seen:
            if made >= n_cf_per_dim:
                break
            m = mutate(sp, dim)
            if m is None:
                continue
            m = replace(m, tid=f"cf_{dim}_{made:02d}")
            if gold_state(m) is None:
                continue
            cf.append(m)
            made += 1
    return seen, cf


# =====================================================================
# 3. Action languages
# =====================================================================

@dataclass
class Instruction:
    name: str
    slots: List[Tuple[str, List[Any]]]
    expand: Any                    # (args, spec) -> list of (primitive, args)


@dataclass
class Language:
    name: str
    instructions: List[Instruction]
    # semantic dimensions this language can name (for E_syntax)
    expressible_dims: FrozenSet[str]
    description: str = ""

    def by_name(self, n: str) -> Optional[Instruction]:
        for i in self.instructions:
            if i.name == n:
                return i
        return None

    def signature_block(self) -> str:
        lines = []
        for ins in self.instructions:
            parts = []
            for sname, vals in ins.slots:
                dom = "|".join(str(v).lower() for v in vals) if len(vals) <= 4 else "..."
                parts.append(f"{sname}:{{{dom}}}")
            lines.append(f"  {ins.name}({', '.join(parts)})")
        lines.append("  STOP()")
        return "\n".join(lines)


def _resolve_prefix(contact: str, pre_resolved: bool):
    return [] if pre_resolved else [("RESOLVE", [contact])]


def _event_block(topic, time, contact, link):
    b = [("CREATE_EVENT", [topic, time, contact])]
    if link:
        b.append(("GET_LINK", [topic, time, contact]))
    return b


def _msg_block(contact, topic, link, channel):
    if channel is None:
        return []
    return [("COMPOSE", [contact, topic, link, channel]),
            ("SEND",    [contact, topic, link, channel])]


def _schedule_notify(contact, time, topic, channel, link, order, pre_resolved):
    prog = _resolve_prefix(contact, pre_resolved)
    ev = _event_block(topic, time, contact, link)
    ms = _msg_block(contact, topic, link, channel)
    if order == "message_first" and ms and not link:
        return prog + ms + ev
    return prog + ev + ms


# ---- L0: raw primitives ---------------------------------------------

def make_L0() -> Language:
    ins = []
    for pname, spec in PRIMITIVES.items():
        ins.append(Instruction(
            name=pname,
            slots=[(s, list(v)) for s, v in spec["slots"]],
            expand=(lambda pn: (lambda args, sp: [(pn, list(args))]))(pname),
        ))
    return Language(
        "L0", ins, frozenset(CF_DIMS),
        "Raw primitives. The model chooses which primitive to call and in what order.",
    )


# ---- L1: typed compositional macros, all optionality preserved -------

def make_L1() -> Language:
    sn = Instruction(
        name="SCHEDULE_AND_NOTIFY",
        slots=[("contact", CONTACTS), ("time", TIMES), ("topic", TOPICS),
               ("channel", CHANNELS + ["none"]), ("include_link", [True, False]),
               ("order", ORDERS)],
        expand=lambda a, sp: _schedule_notify(
            a[0], a[1], a[2], None if a[3] == "none" else a[3], a[4], a[5], sp.pre_resolved),
    )
    so = Instruction(
        name="SCHEDULE_ONLY",
        slots=[("contact", CONTACTS), ("time", TIMES), ("topic", TOPICS)],
        expand=lambda a, sp: _resolve_prefix(a[0], sp.pre_resolved)
        + _event_block(a[2], a[1], a[0], False),
    )
    no = Instruction(
        name="NOTIFY_ONLY",
        slots=[("contact", CONTACTS), ("topic", TOPICS), ("channel", CHANNELS)],
        expand=lambda a, sp: _resolve_prefix(a[0], sp.pre_resolved)
        + _msg_block(a[0], a[1], False, a[2]),
    )
    return Language("L1", [sn, so, no], frozenset(CF_DIMS),
                    "Typed compositional instructions; every distinction is a slot.")


# ---- L2: same instruction count, less optionality --------------------

def make_L2() -> Language:
    sn = Instruction(
        name="SCHEDULE_AND_NOTIFY",
        slots=[("contact", CONTACTS), ("time", TIMES), ("topic", TOPICS),
               ("channel", CHANNELS + ["none"])],
        expand=lambda a, sp: _schedule_notify(
            a[0], a[1], a[2], None if a[3] == "none" else a[3],
            True, "event_first", sp.pre_resolved),   # link and order are fixed
    )
    so = Instruction(
        name="SCHEDULE_ONLY",
        slots=[("contact", CONTACTS), ("time", TIMES), ("topic", TOPICS)],
        expand=lambda a, sp: _resolve_prefix(a[0], sp.pre_resolved)
        + _event_block(a[2], a[1], a[0], False),
    )
    no = Instruction(
        name="NOTIFY_ONLY",
        slots=[("contact", CONTACTS), ("topic", TOPICS), ("channel", CHANNELS)],
        expand=lambda a, sp: _resolve_prefix(a[0], sp.pre_resolved)
        + _msg_block(a[0], a[1], False, a[2]),
    )
    return Language("L2", [sn, so, no],
                    frozenset(CF_DIMS) - {"ordering", "data_dependency"},
                    "Compressed: ordering and link inclusion are baked in.")


# ---- L3: aggressive, removes the channel distinction too -------------

def make_L3() -> Language:
    sn = Instruction(
        name="SCHEDULE_AND_EMAIL",
        slots=[("contact", CONTACTS), ("time", TIMES), ("topic", TOPICS)],
        expand=lambda a, sp: _schedule_notify(
            a[0], a[1], a[2], "email", True, "event_first", sp.pre_resolved),
    )
    so = Instruction(
        name="SCHEDULE_ONLY",
        slots=[("contact", CONTACTS), ("time", TIMES), ("topic", TOPICS)],
        expand=lambda a, sp: _resolve_prefix(a[0], sp.pre_resolved)
        + _event_block(a[2], a[1], a[0], False),
    )
    no = Instruction(
        name="EMAIL_ONLY",
        slots=[("contact", CONTACTS), ("topic", TOPICS)],
        expand=lambda a, sp: _resolve_prefix(a[0], sp.pre_resolved)
        + _msg_block(a[0], a[1], False, "email"),
    )
    return Language("L3", [sn, so, no],
                    frozenset(CF_DIMS) - {"ordering", "data_dependency", "channel"},
                    "Aggressive: channel, ordering and link are all removed.")


# ---- L_rand: matched random-boundary control -------------------------

def make_Lrand(seed: int = 0) -> Language:
    """
    Cut the canonical primitive trace at RANDOM positions instead of at
    dependency boundaries. Matched to L1 on instruction count (3) and on total
    slot count; the boundaries carry no semantic meaning.
    """
    rng = random.Random(seed)
    canonical = ["RESOLVE", "CREATE_EVENT", "GET_LINK", "COMPOSE", "SEND"]
    cuts = sorted(rng.sample(range(1, len(canonical)), 2))
    chunks = [canonical[:cuts[0]], canonical[cuts[0]:cuts[1]], canonical[cuts[1]:]]

    def make_expand(chunk):
        def _e(a, sp):
            out = []
            i = 0
            for pn in chunk:
                k = len(PRIMITIVES[pn]["slots"])
                out.append((pn, list(a[i:i + k])))
                i += k
            return out
        return _e

    ins = []
    for j, chunk in enumerate(chunks):
        slots: List[Tuple[str, List[Any]]] = []
        for pn in chunk:
            for sname, vals in PRIMITIVES[pn]["slots"]:
                slots.append((f"{pn.lower()}_{sname}", list(vals)))
        ins.append(Instruction(name=f"BLOCK_{j}", slots=slots, expand=make_expand(chunk)))
    return Language("L_rand", ins, frozenset(CF_DIMS),
                    "Random-boundary control: real primitives, meaningless grouping.")


def all_languages(seed: int = 0) -> List[Language]:
    return [make_L0(), make_L1(), make_L2(), make_L3(), make_Lrand(seed)]


# =====================================================================
# 4. Oracles: E_syntax and E_semantic
# =====================================================================

def required_dims(spec: TaskSpec) -> FrozenSet[str]:
    """Which semantic dimensions this task actually exercises."""
    d = set()
    if spec.dim != "seen":
        d.add(spec.dim)
    if spec.channel is not None:
        d.add("channel")
    if spec.order == "message_first":
        d.add("ordering")
    if spec.include_link:
        d.add("data_dependency")
    if spec.channel is None:
        d.add("side_effect")
    return frozenset(d)


def e_syntax(lang: Language, spec: TaskSpec) -> bool:
    """Model-independent: can the language NAME every distinction this task needs?"""
    return required_dims(spec) <= lang.expressible_dims


def enumerate_actions(lang: Language, spec: TaskSpec) -> List[Tuple[str, List[Any]]]:
    """
    All (instruction, args) with arguments restricted to the entities relevant to
    this task. Restricting the argument universe is what keeps the oracle search
    tractable; it does not favour any language.
    """
    cand = {
        "contact": [spec.contact], "name": [spec.contact], "attendee": [spec.contact],
        "recipient": [spec.contact], "time": [spec.time], "topic": [spec.topic],
        "channel": CHANNELS + (["none"] if any(s == "channel" and "none" in v
                                               for i in lang.instructions
                                               for s, v in i.slots) else []),
        "include_link": [True, False], "order": ORDERS,
    }
    out: List[Tuple[str, List[Any]]] = []
    for ins in lang.instructions:
        domains = []
        for sname, vals in ins.slots:
            base = sname.split("_")[-1] if sname.startswith(tuple(
                p.lower() for p in PRIMITIVES)) else sname
            dom = cand.get(base, cand.get(sname, vals))
            dom = [v for v in dom if v in vals] or list(vals)
            domains.append(dom)
        for combo in itertools.product(*domains):
            out.append((ins.name, list(combo)))
    return out


def e_semantic(lang: Language, spec: TaskSpec, max_depth: int = 6) -> bool:
    """Model-independent: does ANY sequence in the language reach the gold state?"""
    target = gold_state(spec)
    if target is None:
        return False
    s0 = initial_state(spec)
    if s0 == target:
        return True
    actions = enumerate_actions(lang, spec)
    seen = {s0.key()}
    frontier = deque([(s0, 0)])
    while frontier:
        s, d = frontier.popleft()
        if d >= max_depth:
            continue
        for iname, args in actions:
            ins = lang.by_name(iname)
            if ins is None:
                continue
            try:
                prog = ins.expand(args, spec)
            except Exception:
                continue
            ns = run_program(s, prog)
            if ns is None:
                continue
            if ns == target:
                return True
            k = ns.key()
            if k not in seen:
                seen.add(k)
                frontier.append((ns, d + 1))
    return False


# =====================================================================
# 5. Model scorers
# =====================================================================

class Scorer:
    def score(self, prompt: str, candidates: List[str]) -> List[float]:
        raise NotImplementedError


class DryRunScorer(Scorer):
    """Deterministic pseudo-random scores. Validates the pipeline without a GPU."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, prompt: str, candidates: List[str]) -> List[float]:
        out = []
        for c in candidates:
            h = hash((self.seed, prompt[-200:], c)) % 10_000
            out.append(-5.0 + h / 1000.0)
        return out


class HFScorer(Scorer):
    """Sums token log-probabilities of each candidate continuation."""

    def __init__(self, model_name: str, dtype: str = "auto", device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        td = torch.float16 if (dtype == "auto" and device == "cuda") else (
            torch.float32 if dtype == "auto" else getattr(torch, dtype))

        self.tok = AutoTokenizer.from_pretrained(model_name)
        try:                                   # transformers >= 4.56 renamed this
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=td)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=td)
        self.model = self.model.to(device).eval()
        self.device = device
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        print(f"[HFScorer] {model_name} on {device} ({td})", flush=True)

    @staticmethod
    def _norm(c: str) -> str:
        return " " + c.strip()

    def score(self, prompt: str, candidates: List[str]) -> List[float]:
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
            tot = 0.0
            for pos in range(starts[i], len(s)):
                tot += logprobs[i, pos - 1, batch[i, pos]].item()
            out.append(tot)
        return out


def softmax_entropy(scores: List[float]) -> Tuple[List[float], float, float]:
    """Returns (probs, raw entropy in nats, normalized entropy)."""
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    probs = [e / z for e in exps]
    h_raw = -sum(p * math.log(p) for p in probs if p > 0)
    h_norm = h_raw / math.log(len(probs)) if len(probs) > 1 else 0.0
    return probs, h_raw, h_norm


# =====================================================================
# 6. Decision loop
# =====================================================================

@dataclass
class RunRecord:
    tid: str
    lang: str
    success: bool
    k_struct: int = 0
    k_param: int = 0
    h_tot: float = 0.0
    h_loc_sum: float = 0.0
    n_decisions: int = 0
    prompt_tokens: int = 0
    trace: List[str] = field(default_factory=list)

    @property
    def h_loc(self) -> float:
        return self.h_loc_sum / self.n_decisions if self.n_decisions else 0.0

    @property
    def k_tot(self) -> int:
        return self.k_struct + self.k_param


def build_prompt(lang: Language, spec: TaskSpec, history: List[str], tail: str) -> str:
    return (
        "You are controlling a device by emitting instructions.\n\n"
        f"Available instructions:\n{lang.signature_block()}\n\n"
        f"Address book: {', '.join(CONTACTS)}\n"
        f"Already resolved: {', '.join(sorted(initial_state(spec).resolved)) or 'none'}\n\n"
        f"Task: {utterance(spec)}\n\n"
        "Plan:\n" + ("".join(f"  {h}\n" for h in history) if history else "")
        + tail
    )


def run_task(scorer: Scorer, lang: Language, spec: TaskSpec,
             max_steps: int = 8) -> RunRecord:
    rec = RunRecord(tid=spec.tid, lang=lang.name, success=False)
    state = initial_state(spec)
    history: List[str] = []
    target = gold_state(spec)

    for _ in range(max_steps):
        # ---- structural decision: which instruction (or STOP)
        options = [i.name for i in lang.instructions] + ["STOP"]
        prompt = build_prompt(lang, spec, history, "  ")
        rec.prompt_tokens = max(rec.prompt_tokens, len(prompt) // 4)
        scores = scorer.score(prompt, options)
        _, h_raw, h_norm = softmax_entropy(scores)
        rec.k_struct += 1
        rec.n_decisions += 1
        rec.h_tot += h_raw
        rec.h_loc_sum += h_norm
        choice = options[max(range(len(scores)), key=lambda i: scores[i])]
        if choice == "STOP":
            rec.trace.append("STOP")
            break

        ins = lang.by_name(choice)
        assert ins is not None

        # ---- parametric decisions: one per slot
        args: List[Any] = []
        for sname, vals in ins.slots:
            cand = [str(v).lower() if isinstance(v, bool) else str(v) for v in vals]
            tail = f"  {choice}(" + ", ".join(
                f"{s}={a}" for (s, _), a in zip(ins.slots, args)) + \
                (", " if args else "") + f"{sname}="
            prompt = build_prompt(lang, spec, history, tail)
            scores = scorer.score(prompt, cand)
            _, h_raw, h_norm = softmax_entropy(scores)
            rec.k_param += 1
            rec.n_decisions += 1
            rec.h_tot += h_raw
            rec.h_loc_sum += h_norm
            args.append(vals[max(range(len(scores)), key=lambda i: scores[i])])

        line = f"{choice}(" + ", ".join(
            f"{s}={a}" for (s, _), a in zip(ins.slots, args)) + ")"
        history.append(line)
        rec.trace.append(line)

        try:
            prog = ins.expand(args, spec)
        except Exception:
            break
        ns = run_program(state, prog)
        if ns is None:
            break                      # illegal execution: task fails
        state = ns

    rec.success = (target is not None and state == target)
    return rec


# =====================================================================
# 7. Statistics
# =====================================================================

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def discordance(a_res: Dict[str, bool], b_res: Dict[str, bool]) -> Tuple[int, int, int, int]:
    a = b = c = d = 0
    for tid in a_res:
        if tid not in b_res:
            continue
        x, y = a_res[tid], b_res[tid]
        if x and y:
            a += 1
        elif x and not y:
            c += 1
        elif not x and y:
            b += 1
        else:
            d += 1
    return a, b, c, d


def bootstrap_ci(a_res: Dict[str, bool], b_res: Dict[str, bool],
                 n_boot: int = 10000, seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    tids = [t for t in a_res if t in b_res]
    diffs = []
    for _ in range(n_boot):
        samp = [tids[rng.randrange(len(tids))] for _ in tids]
        da = sum(b_res[t] for t in samp) / len(samp)
        db = sum(a_res[t] for t in samp) / len(samp)
        diffs.append(da - db)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


# =====================================================================
# 8. Main
# =====================================================================

def _harness_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--dry-run", action="store_true", help="no model; validates pipeline")
    ap.add_argument("--n-seen", type=int, default=40)
    ap.add_argument("--n-cf", type=int, default=6, help="counterfactuals per dimension")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="r0_results.json")
    args = ap.parse_args()

    if not args.dry_run and not args.model:
        ap.error("pass --model or --dry-run")

    seen, cf = build_tasks(args.n_seen, args.n_cf, args.seed)
    langs = all_languages(args.seed)
    print(f"tasks: {len(seen)} seen, {len(cf)} counterfactual")
    print(f"languages: {[l.name for l in langs]}\n")

    scorer: Scorer = DryRunScorer(args.seed) if args.dry_run else HFScorer(args.model)

    results: Dict[str, Dict[str, Any]] = {}
    import time
    t_start = time.time()
    for li, lang in enumerate(langs):
        print(f"--- {lang.name} ({li+1}/{len(langs)}) ---", flush=True)
        recs_seen = []
        for i, sp in enumerate(seen):
            recs_seen.append(run_task(scorer, lang, sp))
            if (i + 1) % 10 == 0:
                print(f"    seen {i+1}/{len(seen)}  "
                      f"[{time.time()-t_start:.0f}s elapsed]", flush=True)
        recs_cf = [run_task(scorer, lang, sp) for sp in cf]

        esyn_seen = sum(e_syntax(lang, sp) for sp in seen) / len(seen)
        esem_seen = sum(e_semantic(lang, sp) for sp in seen) / len(seen)
        esyn_cf = sum(e_syntax(lang, sp) for sp in cf) / max(1, len(cf))
        esem_cf = sum(e_semantic(lang, sp) for sp in cf) / max(1, len(cf))

        s_seen = sum(r.success for r in recs_seen) / len(recs_seen)
        s_cf = sum(r.success for r in recs_cf) / max(1, len(recs_cf))
        n = len(recs_seen)

        results[lang.name] = {
            "E_syn_seen": esyn_seen, "E_sem_seen": esem_seen,
            "E_syn_cf": esyn_cf, "E_sem_cf": esem_cf,
            "S_seen": s_seen, "S_cf": s_cf,
            "K_struct": sum(r.k_struct for r in recs_seen) / n,
            "K_param": sum(r.k_param for r in recs_seen) / n,
            "K_tot": sum(r.k_tot for r in recs_seen) / n,
            "H_loc": sum(r.h_loc for r in recs_seen) / n,
            "H_tot": sum(r.h_tot for r in recs_seen) / n,
            "prompt_tokens": sum(r.prompt_tokens for r in recs_seen) / n,
            "per_task_seen": {r.tid: r.success for r in recs_seen},
            "per_task_cf": {r.tid: r.success for r in recs_cf},
            "cf_by_dim": {
                d: (sum(r.success for r, sp in zip(recs_cf, cf) if sp.dim == d)
                    / max(1, sum(1 for sp in cf if sp.dim == d)))
                for d in CF_DIMS
            },
            "example_trace": recs_seen[0].trace if recs_seen else [],
        }
        print(f"  E_syn={esyn_seen:.2f} E_sem={esem_seen:.2f} "
              f"S_seen={s_seen:.2f} S_cf={s_cf:.2f}")
        print(f"  K_struct={results[lang.name]['K_struct']:.1f} "
              f"K_param={results[lang.name]['K_param']:.1f} "
              f"H_tot={results[lang.name]['H_tot']:.2f}")

    # ---- primary endpoint -------------------------------------------
    r0 = results["L0"]["per_task_seen"]
    r1 = results["L1"]["per_task_seen"]
    a, b, c, d = discordance(r0, r1)
    p = mcnemar_exact(b, c)
    lo, hi = bootstrap_ci(r0, r1, seed=args.seed)

    print("\n=== PRIMARY ENDPOINT: L0 vs L1 (paired seen tasks) ===")
    print(f"                 L1 correct   L1 incorrect")
    print(f"  L0 correct        {a:>3}          {c:>3}")
    print(f"  L0 incorrect      {b:>3}          {d:>3}")
    print(f"  exact McNemar p = {p:.4f}   (b={b} fixed, c={c} broken)")
    print(f"  net diff 95% CI = [{lo:+.3f}, {hi:+.3f}]")
    if c > b:
        print("  NOTE: L1 breaks more than it fixes. Inspect the c cell first.")

    results["_primary"] = {"a": a, "b": b, "c": c, "d": d,
                           "mcnemar_p": p, "ci95": [lo, hi]}
    results["_config"] = {"model": args.model or "dry-run", "seed": args.seed,
                          "n_seen": len(seen), "n_cf": len(cf)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n=== TABLE: expressivity and performance ===")
    print(f"{'lang':<8}{'E_syn':>8}{'E_sem':>8}{'S_seen':>8}{'S_cf':>8}"
          f"{'K_str':>8}{'K_par':>8}{'H_tot':>8}")
    for l in langs:
        r = results[l.name]
        print(f"{l.name:<8}{r['E_syn_seen']:>8.2f}{r['E_sem_seen']:>8.2f}"
              f"{r['S_seen']:>8.2f}{r['S_cf']:>8.2f}"
              f"{r['K_struct']:>8.1f}{r['K_param']:>8.1f}{r['H_tot']:>8.2f}")


# harness CLI entry point removed; driven by r3_main() below


# =====================================================================
# PART 2 of 2 -- the r3 experiment runner
# =====================================================================

CKPT = os.path.join(HERE, "r3_ckpt")
os.makedirs(CKPT, exist_ok=True)

PRIMS = ["RESOLVE", "CREATE_EVENT", "GET_LINK", "COMPOSE", "SEND"]
ALL = PRIMS + ["STOP"]
MODEL = "Qwen/Qwen3-0.6B"
SEEDS = [0, 1, 2, 3, 4]
LADDER = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
def save(name, obj):
    with open(os.path.join(CKPT, f"{name}.json"), "w") as fh:
        json.dump(obj, fh, indent=2)


def load(name):
    path = os.path.join(CKPT, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


# --------------------------------------------------------------------------
# constraint families
# --------------------------------------------------------------------------
def successors(prev, pre=False, stop=False, order=False):
    """Three independent constraint families; conjunction is set intersection.

    pre    executor preconditions only -- SEND needs a draft, GET_LINK needs an event
    stop   STOP legal only where a plan could legitimately end
    order  residual sequencing -- no re-resolving, no wandering after a link

    Nothing here is task-specific: it is the dependency structure of the API,
    which any developer writing the tool documentation already knows.
    """
    opts = set(ALL)
    if pre:
        if prev != "COMPOSE":
            opts.discard("SEND")
        if prev != "CREATE_EVENT":
            opts.discard("GET_LINK")
        if prev == "COMPOSE":
            opts &= {"SEND"}
    if stop:
        if prev not in ("CREATE_EVENT", "SEND"):
            opts.discard("STOP")
    if order:
        if prev is not None:
            opts.discard("RESOLVE")
        if prev == "GET_LINK":
            opts &= {"COMPOSE"}
        if prev == "RESOLVE":
            opts &= {"CREATE_EVENT", "COMPOSE", "STOP"}
        if prev == "SEND":
            opts &= {"CREATE_EVENT", "STOP"}
        if prev == "CREATE_EVENT":
            opts &= {"GET_LINK", "COMPOSE", "STOP"}
    return sorted(opts)


def make_lang(name, succ_fn, seed=0):
    """L0's instruction set verbatim plus a successor rule. Nothing is bundled."""
    base = {l.name: l for l in all_languages(seed)}["L0"]
    lang = Language(
        name=name,
        instructions=base.instructions,
        expressible_dims=base.expressible_dims,
        description="raw primitives, restricted successors",
    )
    lang.successors_fn = succ_fn
    return lang


def gold_legal(fn, tasks):
    """A restriction that forbids a correct plan is a bug, not a control."""
    for spec in tasks:
        prev = None
        for name, _ in gold_program(spec):
            if name not in fn(prev):
                return False
            prev = name
        if "STOP" not in fn(prev):
            return False
    return True


# --------------------------------------------------------------------------
# decision loop -- identical to the original except for the successor rule
# --------------------------------------------------------------------------
def run_task_r3(scorer, lang, spec, max_steps=8):
    rec = RunRecord(tid=spec.tid, lang=lang.name, success=False)
    state = initial_state(spec)
    history = []
    target = gold_state(spec)
    fn = getattr(lang, "successors_fn", None)
    prev = None

    for _ in range(max_steps):
        options = list(fn(prev)) if fn else [i.name for i in lang.instructions] + ["STOP"]
        if not options:
            break

        if len(options) == 1:
            choice = options[0]                       # forced: not a decision
        else:
            prompt = build_prompt(lang, spec, history, "  ")
            rec.prompt_tokens = max(rec.prompt_tokens, len(prompt) // 4)
            scores = scorer.score(prompt, options)
            _, h_raw, h_norm = softmax_entropy(scores)
            rec.k_struct += 1
            rec.n_decisions += 1
            rec.h_tot += h_raw
            rec.h_loc_sum += h_norm
            choice = options[max(range(len(scores)), key=lambda i: scores[i])]

        if choice == "STOP":
            rec.trace.append("STOP")
            break

        ins = lang.by_name(choice)
        if ins is None:
            break

        args = []
        for sname, vals in ins.slots:
            cand = [str(v).lower() if isinstance(v, bool) else str(v) for v in vals]
            filled = ", ".join(f"{s}={a}" for (s, _), a in zip(ins.slots, args))
            tail = f"  {choice}(" + filled + (", " if args else "") + f"{sname}="
            scores = scorer.score(build_prompt(lang, spec, history, tail), cand)
            _, h_raw, h_norm = softmax_entropy(scores)
            rec.k_param += 1
            rec.n_decisions += 1
            rec.h_tot += h_raw
            rec.h_loc_sum += h_norm
            args.append(vals[max(range(len(scores)), key=lambda i: scores[i])])

        line = f"{choice}(" + ", ".join(
            f"{s}={a}" for (s, _), a in zip(ins.slots, args)) + ")"
        history.append(line)
        rec.trace.append(line)
        prev = choice

        try:
            prog = ins.expand(args, spec)
        except Exception:
            break
        nxt = run_program(state, prog)
        if nxt is None:
            break
        state = nxt

    rec.success = (target is not None and state == target)
    return rec


def evaluate(scorer, lang, seen, cf, label="", e_syn=None, e_sem=None):
    t0 = time.time()
    rs = []
    for i, spec in enumerate(seen):
        rs.append(run_task_r3(scorer, lang, spec))
        if label and (i + 1) % 20 == 0:
            print(f"      {label} seen {i+1}/{len(seen)} [{time.time()-t0:.0f}s]", flush=True)
    rc = [run_task_r3(scorer, lang, spec) for spec in cf]
    n = len(rs)
    return {
        "S_seen": sum(r.success for r in rs) / n,
        "S_cf": sum(r.success for r in rc) / max(1, len(rc)),
        "K_struct": sum(r.k_struct for r in rs) / n,
        "K_param": sum(r.k_param for r in rs) / n,
        "H_tot": sum(r.h_tot for r in rs) / n,
        "H_loc": sum(r.h_loc for r in rs) / n,
        "prompt_tokens": sum(r.prompt_tokens for r in rs) / n,
        "E_syn": e_syn if e_syn is not None else sum(e_syntax(lang, s) for s in seen) / n,
        "E_sem": e_sem if e_sem is not None else sum(e_semantic(lang, s) for s in seen) / n,
        "per_task_seen": {r.tid: r.success for r in rs},
        "per_task_cf": {r.tid: r.success for r in rc},
        "cf_by_dim": {
            d: (sum(r.success for r, s in zip(rc, cf) if s.dim == d)
                / max(1, sum(1 for s in cf if s.dim == d)))
            for d in CF_DIMS
        },
        "seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------
# part 0 -- predictions
# --------------------------------------------------------------------------
PREDICTIONS = {
    "A_S_L1_mean_across_seeds": 0.80,
    "A_S_L1_min_across_seeds": 0.70,
    "A_S_L0_max_across_seeds": 0.05,
    "A_S_L0tight_mean": 0.15,
    "B_max_S_over_sweep": 0.25,
    "B_corr_S_vs_Htot": -0.30,
    "C_S_precondition_only": 0.12,
    "C_S_stopping_only": 0.05,
    "C_S_ordering_only": 0.05,
    "C_S_all_three": 0.15,
    "D_S_Lrand_matched": 0.12,
    "E_S_L1_1p7B": 0.90,
    "E_S_L0_1p7B": 0.10,
}


def part0():
    if load("prereg_r3"):
        print("  already registered; leaving it alone")
        return
    rec = {
        "author": "[redacted for anonymous review]",
        "written_by_author": True,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions": PREDICTIONS,
    }
    rec["sha256"] = hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode()).hexdigest()
    save("prereg_r3", rec)
    print("  SHA-256:", rec["sha256"])
    print("  Timestamp this hash externally (git commit / OSF / dated email),")
    print("  otherwise the paper must keep saying it has no pre-registration.")


# --------------------------------------------------------------------------
# parts A-E
# --------------------------------------------------------------------------
def partA(scorer):
    res = load("partA") or {}
    for sd in SEEDS:
        if str(sd) in res:
            print(f"  seed {sd} cached")
            continue
        seen, cf = build_tasks(40, 6, sd)
        langs = {l.name: l for l in all_languages(sd)}
        tight = make_lang("L0_tight", lambda p: successors(p, 1, 1, 1), sd)
        assert gold_legal(tight.successors_fn, seen + cf), f"seed {sd}: rule breaks a gold plan"
        row = {}
        for nm, lang in [("L0", langs["L0"]), ("L1", langs["L1"]), ("L0_tight", tight)]:
            row[nm] = evaluate(scorer, lang, seen, cf, label=f"s{sd}/{nm}")
            print(f"  seed {sd} {nm:9s} S={row[nm]['S_seen']:.3f} "
                  f"H={row[nm]['H_tot']:.3f} ({row[nm]['seconds']:.0f}s)", flush=True)
        res[str(sd)] = row
        save("partA", res)

    print("\n  --- across seeds ---")
    for nm in ["L0", "L1", "L0_tight"]:
        v = [res[str(s)][nm]["S_seen"] for s in SEEDS if str(s) in res]
        if v:
            print(f"  {nm:9s} mean={sum(v)/len(v):.3f} min={min(v):.3f} max={max(v):.3f}")
    return res


def partB(scorer, seen, cf):
    req = set()
    for spec in seen + cf:
        prev = None
        for nm, _ in gold_program(spec):
            req.add((prev, nm))
            prev = nm
        req.add((prev, "STOP"))
    opt = [(p, n) for p in [None] + PRIMS for n in ALL if (p, n) not in req]

    rng = random.Random(0)
    res = load("partB") or {}
    for k in [0, 4, 9, 14, 19, len(opt)]:
        nm = f"sweep_{k:02d}"
        if nm in res:
            print(f"  {nm} cached")
            continue
        sub = set(opt) if k >= len(opt) else set(rng.sample(opt, k))
        fn = lambda p, S=sub: sorted({n for n in ALL if (p, n) in req or (p, n) in S})
        assert gold_legal(fn, seen + cf), f"{nm} breaks a gold plan"
        res[nm] = evaluate(scorer, make_lang(nm, fn, 0), seen, cf, label=nm)
        res[nm]["n_optional_edges"] = k
        print(f"  {nm}: H_tot={res[nm]['H_tot']:.3f} S={res[nm]['S_seen']:.3f}", flush=True)
        save("partB", res)

    print("\n  --- sweep (L1 sits at H_tot=2.668 with S=0.875) ---")
    for h, s, nm in sorted((v["H_tot"], v["S_seen"], k) for k, v in res.items()):
        print(f"  H_tot={h:.3f}  S={s:.3f}   {nm}")
    return res


def partC(scorer, seen, cf, quick=False):
    cells = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)] if quick \
        else list(itertools.product([0, 1], repeat=3))
    res = load("partC") or {}
    for pre, stop, order in cells:
        nm = f"pre{pre}_stop{stop}_ord{order}"
        if nm in res:
            print(f"  {nm} cached")
            continue
        fn = lambda p, a=pre, b=stop, c=order: successors(p, a, b, c)
        assert gold_legal(fn, seen + cf), f"{nm} breaks a gold plan"
        res[nm] = evaluate(scorer, make_lang(nm, fn, 0), seen, cf, label=nm)
        res[nm].update(pre=pre, stop=stop, order=order)
        print(f"  {nm}: S={res[nm]['S_seen']:.3f} H_tot={res[nm]['H_tot']:.3f} "
              f"ordering_cf={res[nm]['cf_by_dim']['ordering']:.2f}", flush=True)
        save("partC", res)

    if len(res) == 8:
        print("\n  --- main effects (mean S, family on minus off) ---")
        for fam in ["pre", "stop", "order"]:
            on = [v["S_seen"] for v in res.values() if v[fam] == 1]
            off = [v["S_seen"] for v in res.values() if v[fam] == 0]
            a, b = sum(on) / len(on), sum(off) / len(off)
            print(f"  {fam:6s} on={a:.3f} off={b:.3f} effect={a-b:+.3f}")
    else:
        print(f"\n  {len(res)}/8 cells present; main effects need all 8")

    print("\n  --- all cells, best first (L1 = 0.875) ---")
    for nm, v in sorted(res.items(), key=lambda kv: -kv[1]["S_seen"]):
        print(f"  S={v['S_seen']:.3f}  H_tot={v['H_tot']:.3f}  {nm}")
    return res


def partD(scorer, seen, cf):
    res = load("partD") or {}
    if "L_rand_matched" in res:
        print("  cached")
        return res
    lr = {l.name: l for l in all_languages(0)}["L_rand"]
    short = Language(
        name="L_rand_matched",
        instructions=[
            Instruction(name=i.name,
                        slots=[(f"s{j+1}", v) for j, (_, v) in enumerate(i.slots)],
                        expand=i.expand)
            for i in lr.instructions
        ],
        expressible_dims=lr.expressible_dims,
        description="random boundaries, prompt-matched slot names",
    )
    # renaming slots cannot change what the language can name or reach, so the
    # r0 expressivity values carry over unchanged
    res["L_rand_matched"] = evaluate(scorer, short, seen, cf, label="L_rand_matched",
                                     e_syn=1.0, e_sem=0.4)
    save("partD", res)
    d = res["L_rand_matched"]
    print(f"  S_seen={d['S_seen']:.3f} (L_rand was 0.100, L1 is 0.875)")
    print(f"  prompt_tokens={d['prompt_tokens']:.1f} (L_rand was 339.1, L1 is 236.3)")
    return res


def partE(seen, cf, ladder=None):
    ladder = ladder or LADDER
    res = load("partE") or {}
    langs = {l.name: l for l in all_languages(0)}
    for mn in ladder:
        row = res.get(mn, {})
        if all(k in row for k in ("L0", "L1", "L0_tight")):
            print(f"  {mn} fully cached")
            continue
        try:
            sc = HFScorer(mn)
        except Exception as exc:
            print(f"  SKIPPING {mn}: could not load ({exc})")
            continue
        tight = make_lang("L0_tight", lambda p: successors(p, 1, 1, 1), 0)
        for nm, lang in [("L0", langs["L0"]), ("L1", langs["L1"]),
                         ("L0_tight", tight)]:
            if nm in row:
                print(f"  {mn} {nm} cached")
                continue
            try:
                row[nm] = evaluate(sc, lang, seen, cf,
                                   label=f"{mn.split('/')[-1]}/{nm}")
            except Exception as exc:
                print(f"  {mn} {nm} FAILED mid-run: {exc}")
                break
            res[mn] = row
            save("partE", res)          # checkpoint after EVERY arm
            print(f"  {mn} {nm:9s} S={row[nm]['S_seen']:.3f} "
                  f"H={row[nm]['H_tot']:.3f} ({row[nm]['seconds']:.0f}s)",
                  flush=True)
        del sc
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n  --- ladder so far ---")
    print(f"  {'model':<20}{'L0':>8}{'L0_tight':>10}{'L1':>8}")
    for mn, r in res.items():
        print(f"  {mn.split('/')[-1]:<20}"
              + "".join(f"{r[k]['S_seen']:>8.3f}" if k in r else f"{'--':>8}"
                        for k in ("L0", "L0_tight", "L1")))
    return res



# --------------------------------------------------------------------------
def r3_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["0", "C", "B", "D", "A", "E"],
                    help="which parts to run, in order (default: all, C first)")
    ap.add_argument("--quick", action="store_true",
                    help="part C with 4 cells instead of 8")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--models", nargs="+", default=None,
                    help="ladder models for part E (default: 0.6B 1.7B 4B)")
    args = ap.parse_args()

    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"CUDA available: {gpu}"
              + (f"  ({torch.cuda.get_device_name(0)})" if gpu else ""))
        if not gpu:
            print("\nWARNING: no GPU detected. This will run, but very slowly.\n"
                  "         On CPU a single arm can take an hour or more.\n")
    except ImportError:
        sys.exit("ERROR: torch is not installed.  pip install torch transformers accelerate")

    seen, cf = build_tasks(40, 6, 0)
    print(f"tasks: {len(seen)} seen, {len(cf)} counterfactual")

    scorer = None
    t0 = time.time()
    for part in args.parts:
        print(f"\n{'='*66}\nPART {part}\n{'='*66}", flush=True)
        if part == "0":
            part0()
            continue
        if part == "E":
            partE(seen, cf, ladder=args.models)
            continue
        if scorer is None:
            scorer = HFScorer(args.model)
        if part == "A":
            partA(scorer)
        elif part == "B":
            partB(scorer, seen, cf)
        elif part == "C":
            partC(scorer, seen, cf, quick=args.quick)
        elif part == "D":
            partD(scorer, seen, cf)
        else:
            print(f"  unknown part {part!r}, skipping")

    commit = "unknown"
    try:
        commit = getattr(scorer.model.config, "_commit_hash", "unknown")
    except AttributeError:
        pass

    out = {
        "_config": {
            "model": args.model,
            "commit_hash": commit,
            "seeds": SEEDS,
            "n_seen": len(seen),
            "n_cf": len(cf),
            "max_steps": 8,
            "decoding": "enumerated candidate scoring, argmax",
            "parts_requested": args.parts,
            "quick": args.quick,
        },
        "_prereg": load("prereg_r3"),
    }
    for name, key in [("partA", "seeds"), ("partB", "entropy_sweep"),
                      ("partC", "constraint_factorial"), ("partD", "prompt_matched"),
                      ("partE", "model_ladder")]:
        val = load(name)
        if val:
            out[key] = val

    with open(os.path.join(HERE, "r3_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'='*66}")
    print(f"wrote r3_results.json  ({time.time()-t0:.0f}s total)")
    print("parts included:", [k for k in out if not k.startswith("_")])
    print("Send that one file back.")


if __name__ == "__main__":
    r3_main()

# adapter_extra.py -- wires run_extra.py to the released bundle.
# Everything comes from r3_all_in_one.py, which is self-contained: it carries the
# executor, task generator, all grammars, the released decision loop (run_task)
# and the successor-rule variant (run_task_r3). No cross-module dataclasses.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _ilu

def _load(modname, fname):
    spec = _ilu.spec_from_file_location(
        modname, os.path.join(os.path.dirname(os.path.abspath(__file__)), fname))
    m = _ilu.module_from_spec(spec); sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m

B = _load("r3_all_in_one", "r3_all_in_one.py")

_SEED = {"value": 0}


class _WrapScorer(B.Scorer):
    """Adapts run_extra's already-loaded HF (model, tok) pair to the released
    Scorer interface, delegating to the released scoring code verbatim."""
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        try:                      # torch only needed for the real scoring path
            import torch
            self.torch = torch
            self.device = str(next(model.parameters()).device)
        except ImportError:
            self.torch, self.device = None, "cpu"

    def score(self, prompt, candidates):
        return B.HFScorer.score(self, prompt, candidates)


def run_task_bundle(model, tok, language_name: str, task: dict) -> dict:
    spec = task["task"]
    scorer = _WrapScorer(model, tok)
    if language_name == "L0p":
        lang = B.make_lang("L0_tight", lambda p: B.successors(p, 1, 1, 1), _SEED["value"])
        rec = B.run_task_r3(scorer, lang, spec)
    else:
        lang = B.make_L0() if language_name == "L0" else B.make_L1()
        rec = B.run_task(scorer, lang, spec)
    return dict(success=bool(rec.success), htot=float(rec.h_tot),
                kstruct=int(rec.k_struct), kparam=int(rec.k_param),
                n_instr=int(len([t for t in rec.trace if t != "STOP"])),
                halted_illegal=bool(getattr(rec, "halted_illegal", False)),
                history=list(rec.trace))


def load_tasks(seed: int):
    _SEED["value"] = seed
    seen, cf = B.build_tasks(seed=seed)
    out = [dict(id=sp.tid, split="seen", cf_dim=None, source_id=None, task=sp) for sp in seen]
    out += [dict(id=sp.tid, split="cf", cf_dim=getattr(sp, "cf_dim", None),
                 source_id=getattr(sp, "source_id", None), task=sp) for sp in cf]
    assert len(out) == 70, len(out)
    return out


def wire(run_extra_module):
    """Call once after importing run_extra: installs RUN_TASK and load_tasks."""
    run_extra_module.RUN_TASK = run_task_bundle
    run_extra_module.load_tasks = load_tasks
    # load_executor/load_language never fire when RUN_TASK is set, but make them
    # inert rather than raising if run_extra's plumbing probes them:
    run_extra_module.load_executor = lambda: None
    run_extra_module.load_language = lambda name: None
    return run_extra_module

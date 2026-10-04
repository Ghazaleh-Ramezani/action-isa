#!/usr/bin/env python3
"""
bfcl_experiment.py — run the paper's decision loop on the BFCL v4 multi-turn
subset produced by bfcl_recipe_harness.py.

  python bfcl_experiment.py --scorer gold          # executor/oracle check (no GPU)
  python bfcl_experiment.py --scorer random --seed 0
  python bfcl_experiment.py --scorer hf --model Qwen/Qwen3-0.6B

Decoding is identical to the paper: at each step the legal candidate set is
ENUMERATED and scored, argmax is executed, no free generation, so truncation
is impossible and p_k over the legal set is exact. Success is exact
final-state agreement with the ground-truth program's final state.
"""
import argparse, copy, itertools, json, math, os, re, sys, time, importlib.util
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

APIS = {'GorillaFileSystem': ('gorilla_file_system','GorillaFileSystem'),
        'VehicleControlAPI': ('vehicle_control','VehicleControlAPI'),
        'TradingBot': ('trading_bot','TradingBot')}
STOP = ('stop',)
MAX_STEPS_SLACK = 3          # budget = len(gold) + slack, mirrors the paper's 8 vs 5



def load_api(name, cls):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def _src(node):
    import ast
    return repr(ast.literal_eval(node))


def parse_call(call):
    """(fname, {slot: literal_source}) via Python's own parser, so escaped
    quotes and commas inside string literals are handled exactly. Positional
    arguments are named by position (_0, _1, ...)."""
    import ast
    try:
        node = ast.parse(call.strip(), mode='eval').body
        fname = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
        args = {}
        for i, a in enumerate(node.args):
            args['_%d' % i] = _src(a)
        for kw in node.keywords:
            args[kw.arg] = _src(kw.value)
        return fname, args
    except Exception:
        return call.split('(')[0], {}


def render_call(fname, args):
    """Inverse of parse_call: positional slots (_i) first, then keywords."""
    pos = sorted((k for k in args if k.startswith('_')), key=lambda k: int(k[1:]))
    parts = [args[k] for k in pos]
    parts += ['%s=%s' % (k, args[k]) for k in args if not k.startswith('_')]
    return fname + '(' + ','.join(parts) + ')'


def snapshot(inst):
    return {k: copy.deepcopy(v) for k, v in vars(inst).items() if not k.startswith('_')}


def states_equal(a, b):
    return repr(sorted(a.items())) == repr(sorted(b.items()))


# ------------------------------------------------------------------ languages
def sig_of(calls):
    sigs = {}
    for c in calls:
        f, a = parse_call(c)
        sigs.setdefault(f, list(a.keys()))
    return sigs


def candidates(rec, bundles, lang, cap=64, substitutive=False):
    """Candidate actions. 'call' = one API call; 'bundle' = two calls as one
    instruction (L1 only). In substitutive mode L1 REPLACES the bundled
    primitives instead of adding to them, which is the recoding the sandbox
    experiment performs; it is legal only where neither part occurs outside
    the bundle in this task's gold (rec['substitutive'])."""
    sigs, U = sig_of(rec['gold_l0']), rec['universe']
    drop = set()
    if lang == 'L1' and substitutive and rec.get('substitutive'):
        for b in bundles:
            if b['cls'] == rec['cls'] and b['name'] in rec.get('removable', []):
                drop.update(b['parts'])

    def calls_for(f, slots):
        doms = [U.get(s, []) for s in slots]
        if any(len(d) == 0 for d in doms):
            return []
        out = []
        for combo in itertools.product(*doms):
            out.append(render_call(f, dict(zip(slots, combo))))
            if len(out) >= cap:
                break
        return out

    acts = []
    for f, slots in sigs.items():
        if f in drop:
            continue
        acts += [('call', c) for c in calls_for(f, slots)]
    if lang == 'L1':
        for b in bundles:
            if b['cls'] != rec['cls']:
                continue
            f, g = b['parts']
            if f not in sigs or g not in sigs:
                continue
            for c1 in calls_for(f, sigs[f]):
                for c2 in calls_for(g, sigs[g]):
                    acts.append(('bundle', b['name'], [c1, c2]))
    return acts


def render(act):
    if act[0] == 'call':
        # normalise through the parser so gold strings and enumerated
        # candidates have byte-identical surface form
        f, a = parse_call(act[1])
        return render_call(f, a) if a or '()' in act[1] else act[1]
    if act[0] == 'bundle':
        # Every distinction the bundle removes must survive as its own typed
        # slot (Corollary 1): when two parts share a slot name, qualify it by
        # part index so distinct programs never render identically.
        _, name, calls = act
        per = [parse_call(c)[1] for c in calls]
        seen = {}
        for a in per:
            for k in a:
                seen[k] = seen.get(k, 0) + 1
        args = {}
        for j, a in enumerate(per):
            for k, v in a.items():
                key = k if seen[k] == 1 and not k.startswith('_') else '%s%d' % (k.lstrip('_') or 'arg', j + 1)
                args[key] = v
        return render_call(name, args)
    return 'STOP()'


def act_calls(act):
    if act[0] == 'call':
        return [act[1]]
    if act[0] == 'bundle':
        return act[2]
    return []


# -------------------------------------------------------------------- scorers
class GoldScorer:
    """Executor/oracle check: always prefers the next gold action. If the
    harness is sound this reaches S = 1.0 for every task in both languages."""
    name = 'gold'

    def __init__(self, **kw):
        self.plan = None

    def set_plan(self, plan):
        self.plan = list(plan)
        self.i = 0

    def score(self, prompt, acts):
        want = render(self.plan[self.i]) if self.i < len(self.plan) else 'STOP()'
        return [0.0 if render(a) == want else -50.0 for a in acts]

    def advance(self):
        self.i += 1


class RandomScorer:
    name = 'random'

    def __init__(self, seed=0, **kw):
        import random
        self.rng = random.Random(seed)

    def set_plan(self, plan):
        pass

    def score(self, prompt, acts):
        return [self.rng.random() for _ in acts]

    def advance(self):
        pass


class HFScorer:
    """Length-normalised log-likelihood of each rendered candidate, given the
    prompt. Enumerated scoring, argmax, no sampling."""
    name = 'hf'

    def __init__(self, model='Qwen/Qwen3-0.6B', dtype='auto', **kw):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        td = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=td)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=td)
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.dev).eval()
        print('[HFScorer] %s on %s (%s)' % (model, self.dev, td), flush=True)

    def set_plan(self, plan):
        pass

    def advance(self):
        pass

    def score(self, prompt, acts):
        torch = self.torch
        p_ids = self.tok(prompt, return_tensors='pt').input_ids[0]
        out = []
        with torch.no_grad():
            for a in acts:
                c_ids = self.tok(render(a), return_tensors='pt',
                                 add_special_tokens=False).input_ids[0]
                ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(self.dev)
                logits = self.model(ids).logits[0].float()
                lp = torch.log_softmax(logits[:-1], -1)
                tgt = ids[0][1:]
                start = len(p_ids) - 1
                tok_lp = lp[start:, :].gather(1, tgt[start:].unsqueeze(1)).squeeze(1)
                out.append((tok_lp.sum() / max(1, len(c_ids))).item())
        return out


SCORERS = {'gold': GoldScorer, 'random': RandomScorer, 'hf': HFScorer}


# ---------------------------------------------------------------- decision loop
def build_prompt(rec, lang, bundles, history, turns=None):
    if turns is None:
        turns = [m['content'] for turn in rec['question'] for m in turn if m['role'] == 'user']
    else:
        turns = [c for t in turns for c in t]
    lines = ['You control an API. Emit one instruction at a time.',
             'Task:'] + ['  ' + t for t in turns]
    if lang == 'L1':
        my = [b for b in bundles if b['cls'] == rec['cls']]
        if my:
            lines.append('Compound instructions available: ' +
                         ', '.join('%s (= %s then %s)' % (b['name'], *b['parts']) for b in my))
    if history:
        lines.append('Executed so far: ' + '; '.join(history))
    lines.append('Next instruction:')
    return '\n'.join(lines) + ' '


def run_turnwise(rec, bundles, lang, scorer, final_state, substitutive=False):
    """BFCL's own protocol: user turns arrive one at a time and the episode
    for each turn is short. Stopping is then a 1-3 step decision rather than a
    6-9 step one, which is the failure mode that dominated the flat run."""
    acts = candidates(rec, bundles, lang, substitutive=substitutive) + [STOP]
    cls = rec['cls']
    inst = CLS[cls]()
    inst._load_scenario(copy.deepcopy(rec['initial_config'].get(cls, {})))
    turns = [[m['content'] for m in t if m['role'] == 'user'] for t in rec['question']]
    gturns = rec['gold_turns']
    history, ent, nstruct = [], [], 0
    for ti, gt_turn in enumerate(gturns):
        if hasattr(scorer, 'set_plan'):
            plan = [('call', c) for c in gt_turn]
            if lang == 'L1':
                plan = []
                i = 0
                while i < len(gt_turn):
                    hit = None
                    if i + 1 < len(gt_turn):
                        f = gt_turn[i].split('(')[0]
                        g = gt_turn[i + 1].split('(')[0]
                        for b in bundles:
                            if b['cls'] == cls and b['parts'] == [f, g]:
                                hit = b
                                break
                    if hit:
                        plan.append(('bundle', hit['name'], [gt_turn[i], gt_turn[i + 1]]))
                        i += 2
                    else:
                        plan.append(('call', gt_turn[i]))
                        i += 1
            scorer.set_plan(plan)
        budget = len(gt_turn) + 1
        seen = turns[:ti + 1] if ti < len(turns) else turns
        for _ in range(budget):
            prompt = build_prompt(rec, lang, bundles, history, turns=seen)
            s = scorer.score(prompt, acts)
            m = max(s)
            ex = [math.exp(v - m) for v in s]
            Z = sum(ex)
            p = [e / Z for e in ex]
            ent.append(-sum(pi * math.log(pi) for pi in p if pi > 0))
            act = acts[s.index(m)]
            nstruct += 1
            if hasattr(scorer, 'advance'):
                scorer.advance()
            if act == STOP:
                break
            for call in act_calls(act):
                try:
                    fn = getattr(inst, call.split('(')[0])
                    eval('fn(%s)' % call.split('(', 1)[1][:-1], {'fn': fn})
                except Exception:
                    pass
            history.append(render(act))
    gold_state = final_state(rec, rec['gold_l0'])
    return dict(success=int(states_equal(snapshot(inst), gold_state)),
                H_tot=sum(ent), steps=len(history), K_struct=nstruct, K_param=0)


def run_task(rec, bundles, lang, scorer, final_state, oracle_stop=False,
             substitutive=False):
    # oracle_stop grants the stopping decision: the step budget is exactly the
    # gold length and STOP is removed from the candidate set, so the arm is
    # scored on selection and ordering alone. This mirrors the paper's factorial
    # treatment of stopping as a separate constraint family.
    acts = candidates(rec, bundles, lang, substitutive=substitutive) + \
        ([] if oracle_stop else [STOP])
    gold = rec['gold_l0'] if lang == 'L0' else \
        [('bundle', s['bundle'], s['calls']) if 'bundle' in s else ('call', s['call'])
         for s in rec['gold_l1']]
    gold_plan = [('call', c) for c in rec['gold_l0']] if lang == 'L0' else gold
    if hasattr(scorer, 'set_plan'):
        scorer.set_plan(gold_plan)

    cls = rec['cls']
    inst = CLS[cls]()
    inst._load_scenario(copy.deepcopy(rec['initial_config'].get(cls, {})))
    history, ent, nstruct, nparam = [], [], 0, 0
    budget = (len(rec['gold_l1']) if lang == 'L1' else len(rec['gold_l0'])) \
        if oracle_stop else len(rec['gold_l0']) + MAX_STEPS_SLACK
    for _ in range(budget):
        prompt = build_prompt(rec, lang, bundles, history)
        s = scorer.score(prompt, acts)
        m = max(s)
        ex = [math.exp(v - m) for v in s]
        Z = sum(ex)
        p = [e / Z for e in ex]
        ent.append(-sum(pi * math.log(pi) for pi in p if pi > 0))
        k = s.index(m)
        act = acts[k]
        nstruct += 1
        if act[0] == 'call':
            nparam += len(parse_call(act[1])[1])
        elif act[0] == 'bundle':
            nparam += len(set().union(*(parse_call(c)[1].keys() for c in act[2])))
        if hasattr(scorer, 'advance'):
            scorer.advance()
        if act == STOP:
            break
        for call in act_calls(act):
            try:
                fn = getattr(inst, call.split('(')[0])
                eval('fn(%s)' % call.split('(', 1)[1][:-1], {'fn': fn})
            except Exception:
                pass
        history.append(render(act))
    reached = snapshot(inst)
    gold_state = final_state(rec, rec['gold_l0'])
    return dict(success=int(states_equal(reached, gold_state)),
                H_tot=sum(ent), steps=len(history),
                K_struct=nstruct, K_param=nparam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scorer', default='gold', choices=list(SCORERS))
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--subset', default=os.path.join(HERE, 'bfcl_subset.json'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--oracle-stop', action='store_true',
                    help='grant the stopping decision: budget = gold length, no STOP action')
    ap.add_argument('--substitutive', action='store_true',
                    help='L1 REPLACES bundled primitives (sandbox-style recoding); '
                         'implies --only-substitutive since it is illegal elsewhere')
    ap.add_argument('--turnwise', action='store_true',
                    help="run BFCL's own per-turn protocol instead of one flat episode")
    args = ap.parse_args()

    global CLS
    CLS = {k: load_api(*v) for k, v in APIS.items()}

    def final_state(rec, calls):
        cls = rec['cls']
        inst = CLS[cls]()
        inst._load_scenario(copy.deepcopy(rec['initial_config'].get(cls, {})))
        for call in calls:
            fn = getattr(inst, call.split('(')[0])
            eval('fn(%s)' % call.split('(', 1)[1][:-1], {'fn': fn})
        return snapshot(inst)

    D = json.load(open(args.subset))
    bundles, records = D['bundles'], D['records']
    if args.substitutive:
        records = [r for r in records if r.get('substitutive')]
        print('substitutive mode: %d tasks where L1 may replace its parts' % len(records))
    scorer = SCORERS[args.scorer](model=args.model, seed=args.seed)

    res = {}
    t0 = time.time()
    for lang in ('L0', 'L1'):
        rows = {}
        for rec in records:
            if args.turnwise:
                rows[rec['id']] = run_turnwise(rec, bundles, lang, scorer, final_state,
                                               substitutive=args.substitutive)
            else:
                rows[rec['id']] = run_task(rec, bundles, lang, scorer, final_state,
                                           oracle_stop=args.oracle_stop,
                                           substitutive=args.substitutive)
        n = len(rows)
        res[lang] = dict(
            S=sum(r['success'] for r in rows.values()) / n,
            H_tot=sum(r['H_tot'] for r in rows.values()) / n,
            K_struct=sum(r['K_struct'] for r in rows.values()) / n,
            K_param=sum(r['K_param'] for r in rows.values()) / n,
            per_task={k: v['success'] for k, v in rows.items()})
        print('%s  S=%.3f  H_tot=%.3f  K_struct=%.2f  (%.0fs)' %
              (lang, res[lang]['S'], res[lang]['H_tot'], res[lang]['K_struct'],
               time.time() - t0), flush=True)

    a = res['L0']['per_task']
    b = res['L1']['per_task']
    ks = sorted(a)
    tab = dict(a=sum(1 for k in ks if a[k] and b[k]),
               b=sum(1 for k in ks if not a[k] and b[k]),
               c=sum(1 for k in ks if a[k] and not b[k]),
               d=sum(1 for k in ks if not a[k] and not b[k]))
    n = tab['b'] + tab['c']
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(tab['b'], tab['c']) + 1)) / 2 ** n) if n else 1.0
    res['_mcnemar'] = dict(tab, p=p)
    res['_config'] = dict(scorer=args.scorer, model=args.model if args.scorer == 'hf' else None,
                          seed=args.seed, n_tasks=len(records), oracle_stop=args.oracle_stop,
                          substitutive=args.substitutive, turnwise=args.turnwise,
                          decoding='enumerated candidate scoring, argmax',
                          success='exact final-state agreement with ground truth',
                          bundles=[x['name'] for x in bundles])
    print('McNemar L0 vs L1: a=%(a)d b=%(b)d c=%(c)d d=%(d)d  p=%(p)s' % res['_mcnemar'])
    tags = ('_sub' if args.substitutive else '') + ('_turn' if args.turnwise else '') + \
        ('_ostop' if args.oracle_stop else '')
    out = args.out or os.path.join(HERE, 'bfcl_results_%s%s.json' % (args.scorer, tags))
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()

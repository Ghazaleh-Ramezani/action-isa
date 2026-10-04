#!/usr/bin/env python3
"""
bfcl_recipe_harness.py — apply the paper's recoding recipe to a BFCL v4
multi-turn subset with a final-state oracle.

Deterministic parts (this file, no GPU):
  1. subset: single-class tasks on {GorillaFileSystem, VehicleControlAPI,
     TradingBot} whose ground truth executes cleanly and changes state.
  2. per-task closed entity universe: literal argument values of the task's
     ground truth (the sandbox analogue: candidates are enumerable per task).
  3. dependency mining: adjacent GT calls (f,g) are dependent iff swapping
     them changes the final state or raises; function-level bigrams with
     dep>=2 and dep>indep become bundles.
  4. languages:
       L0 = the API exactly as written (single calls over the task universe)
       L1 = L0 + mined bundles, each bundle one instruction whose slots are
            the union of the parts' slots (removed sequencing distinctions
            become typed slots)
  5. expressivity: for every task, the gold program is expressible in both
     languages (L0 trivially; L1 by greedy left-to-right bundling of gold),
     and the oracle certifies the bundled form reaches the same final state.

GPU part (Colab): enumerated candidate scoring, argmax, exactly the paper's
decision loop; see run_bfcl_colab.ipynb.
"""
import json, copy, re, sys, os, importlib.util
from collections import defaultdict

# the downloaded API modules import from a bfcl_eval package that ships
# beside them; make that stub importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APIS = {'GorillaFileSystem': ('gorilla_file_system','GorillaFileSystem'),
        'VehicleControlAPI': ('vehicle_control','VehicleControlAPI'),
        'TradingBot': ('trading_bot','TradingBot')}

def load_api(name, cls, base='.'):
    spec = importlib.util.spec_from_file_location(name, f'{base}/{name}.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls)

def snapshot(inst):
    return {k: copy.deepcopy(v) for k,v in vars(inst).items() if not k.startswith('_')}

def states_equal(a,b):
    return repr(sorted(a.items())) == repr(sorted(b.items()))

def make_runner(CLS):
    def final_state(task, calls):
        cls = task['involved_classes'][0]
        inst = CLS[cls]()
        inst._load_scenario(copy.deepcopy(task.get('initial_config',{}).get(cls,{})))
        for call in calls:
            fn = getattr(inst, call.split('(')[0])
            eval(f"fn({call.split('(',1)[1][:-1]})", {'fn':fn})
        return snapshot(inst)
    return final_state



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

def main(datadir='.', out='bfcl_subset.json'):
    CLS = {k: load_api(*v, base=datadir) for k,v in APIS.items()}
    final_state = make_runner(CLS)
    tasks=[json.loads(l) for l in open(f'{datadir}/multi_turn_base.json')]
    gts={json.loads(l)['id']: json.loads(l)['ground_truth']
         for l in open(f'{datadir}/gt_multi_turn_base.json')}

    # 1. subset
    subset=[]
    for t in tasks:
        if len(t['involved_classes'])!=1 or t['involved_classes'][0] not in APIS: continue
        gt=[c for turn in gts[t['id']] for c in turn]
        try:
            s_full = final_state(t, gt); s_init = final_state(t, [])
        except Exception:
            continue
        if states_equal(s_full, s_init): continue
        subset.append((t, gt, s_full))
    print(f'subset: {len(subset)} tasks')

    # 3. dependency mining (function-level, corpus-wide)
    dep = defaultdict(lambda: [0,0])
    for t, gt, s_gold in subset:
        cls=t['involved_classes'][0]
        for i in range(len(gt)-1):
            f,g = gt[i].split('(')[0], gt[i+1].split('(')[0]
            sw = gt[:i]+[gt[i+1],gt[i]]+gt[i+2:]
            try:
                s = final_state(t, sw); d = not states_equal(s, s_gold)
            except Exception:
                d = True
            dep[(cls,(f,g))][int(not d)] += 1
    bundles=[]
    for (cls,(f,g)),(d,i) in dep.items():
        if d>=2 and d>i and f!=g:
            bundles.append({'cls':cls,'parts':[f,g],
                            'name':f'{f.upper()}_THEN_{g.upper()}'})
    print(f'bundles: {[b["name"] for b in bundles]}')

    # 2+4+5. per-task record: universe, L0/L1 candidate spec, gold in both languages
    records=[]
    for t, gt, s_gold in subset:
        cls=t['involved_classes'][0]
        # closed universe: values by slot name
        universe=defaultdict(set)
        parsed=[parse_call(c) for c in gt]
        for fname,args in parsed:
            for k,v in args.items(): universe[k].add(v)
        # gold in L1: greedy left-to-right bundling
        my_bundles=[b for b in bundles if b['cls']==cls]
        gold_l1=[]; i=0
        while i < len(gt):
            hit=None
            if i+1 < len(gt):
                f,g = gt[i].split('(')[0], gt[i+1].split('(')[0]
                for b in my_bundles:
                    if b['parts']==[f,g]: hit=b; break
            if hit:
                gold_l1.append({'bundle':hit['name'],'calls':[gt[i],gt[i+1]]}); i+=2
            else:
                gold_l1.append({'call':gt[i]}); i+=1
        # oracle certification: bundled gold reaches gold state
        flat=[c for step in gold_l1 for c in (step.get('calls') or [step['call']])]
        assert states_equal(final_state(t, flat), s_gold), t['id']
        # per-turn gold (BFCL feeds user turns incrementally; a flat episode
        # makes stopping a 6-9 step decision instead of the 1-3 step one the
        # benchmark actually poses)
        gt_turns = gts[t['id']]
        # substitutability: a bundle may REPLACE its parts only if neither part
        # ever occurs outside the bundle in this task's gold (else E_syn drops)
        fns = [c.split('(')[0] for c in gt]
        def outside(f, g):
            for i, fn in enumerate(fns):
                if fn == f and not (i + 1 < len(fns) and fns[i + 1] == g):
                    return True
                if fn == g and not (i > 0 and fns[i - 1] == f):
                    return True
            return False
        used = [b for b in my_bundles
                if any(s.get('bundle') == b['name'] for s in gold_l1)]
        # A bundle whose two parts fall in DIFFERENT user turns cannot be
        # emitted as one instruction under the benchmark's own turn protocol,
        # so removing its parts would make the gold inexpressible there.
        def straddles(f, g):
            for turn in gt_turns:
                tf = [c.split('(')[0] for c in turn]
                for i, fn in enumerate(tf):
                    if fn == f and i + 1 < len(tf) and tf[i + 1] == g:
                        return False      # adjacent inside some turn: fine
            return True                   # never adjacent within a turn
        substitutive = bool(used) and all(
            not outside(*b['parts']) and not straddles(*b['parts']) for b in used)
        records.append({'id':t['id'],'cls':cls,
                        'initial_config':t.get('initial_config',{}),
                        'question':t['question'],
                        'gold_l0':gt,
                        'gold_turns':gt_turns,
                        'gold_l1':gold_l1,
                        'substitutive':substitutive,
                        'removable':[b['name'] for b in used] if substitutive else [],
                        'universe':{k:sorted(v) for k,v in universe.items()}})
    json.dump({'bundles':bundles,'records':records}, open(out,'w'), indent=1)
    n_b=sum(1 for r in records if any('bundle' in s for s in r['gold_l1']))
    print(f'wrote {out}: {len(records)} tasks, gold-L1 uses a bundle in {n_b}')
    print('expressivity: gold expressible and state-certified in BOTH languages, all tasks')

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '.')

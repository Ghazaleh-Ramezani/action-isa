#!/usr/bin/env bash
# run_extra.sh -- the two remaining runs, on your own machine.
set -e
python3 -c "import torch,transformers" 2>/dev/null || pip install -q -r requirements-extra.txt

python3 - <<'EOF'
import importlib.util as ilu, sys
sys.path.insert(0,'.')
sp=ilu.spec_from_file_location('run_extra','run_extra.py'); RE=ilu.module_from_spec(sp)
sys.modules['run_extra']=RE; sp.loader.exec_module(RE)
import adapter_extra as A; A.wire(RE)
sys.argv=['run_extra.py','--selftest']
try: RE.main()
except SystemExit: pass
EOF

for PLAN in family ladder; do
python3 - "$PLAN" <<'EOF'
import importlib.util as ilu, sys
sys.path.insert(0,'.')
sp=ilu.spec_from_file_location('run_extra','run_extra.py'); RE=ilu.module_from_spec(sp)
sys.modules['run_extra']=RE; sp.loader.exec_module(RE)
import adapter_extra as A; A.wire(RE)
sys.argv=['run_extra.py','--plan',sys.argv[1],'--go']
try: RE.main()
except SystemExit: pass
EOF
done

python3 extra_analysis.py results/extra --predictions predictions.json
python3 check_anonymity.py .

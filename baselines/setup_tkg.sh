#!/bin/bash
# Patch official RE-GCN / TiRGN / HiSMatch: custom dataset names, seed from env, per-seed model file. Links data folders.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
set -e
cd $FINCHRONO_BASE
for r in RE-GCN TiRGN HiSMatch; do
  grep -q "'findkg'" $r/rgcn/utils.py || sed -i "s/dataset in \['ICEWS18',/dataset in ['findkg', 'gdeltfin', 'ICEWS18',/" $r/rgcn/utils.py
  mkdir -p $r/data $r/models
  for d in findkg gdeltfin; do [ -d $FINCHRONO_OUT/tkg/$d ] && ln -sfn $FINCHRONO_OUT/tkg/$d $r/data/$d; done
  for f in $r/src/main.py; do
    grep -q FC_SEED_PATCH $f || sed -i '0,/^import torch/s//import torch  # FC_SEED_PATCH\nimport os as _o, random as _rnd, numpy as _np\n_s = int(_o.environ.get("SEED", 0)); torch.manual_seed(_s); torch.cuda.manual_seed_all(_s); _np.random.seed(_s); _rnd.seed(_s)/' $f
  done
done
for r in RE-GCN TiRGN HiSMatch; do
  grep -q FC_MODEL_PATCH $r/src/main.py || sed -i "s#^\( *\)model_state_file = \(.*\)\$#\1model_state_file = \2 + '-seed' + _o.environ.get('SEED', '0')  \# FC_MODEL_PATCH#" $r/src/main.py
done
# official get_repetitive_history.py uses helpers it never imports: get_data_with_t lives in its utils.py, get_big_graph in get_history_dict.py
python3 - <<'PY'
import re
p = 'HiSMatch/hismatch_preprocess/get_repetitive_history.py'; s = open(p).read()
if 'FC_PATCH' not in s:
    d = open('HiSMatch/hismatch_preprocess/get_history_dict.py').read()
    m = re.search(r'^def get_big_graph\(data, num_rels\):\n(?:(?:    .*)?\n)+', d, re.M)
    s = re.sub(r'^from utils import .*$', 'from utils import get_total_number, load_quadruples, get_data_with_t  # FC_PATCH', s, count=1, flags=re.M)
    s = s.replace('graph_dict_train = {}', m.group(0) + '\ngraph_dict_train = {}', 1)
    open(p, 'w').write(s)
PY
echo patched

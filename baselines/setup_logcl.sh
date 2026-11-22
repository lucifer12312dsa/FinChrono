#!/bin/bash
# setup_logcl.sh <dataset>: LogCL official repo on our split. No word graph (entity names here are not ICEWS-style), so no --add-static-graph.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
cd $FINCHRONO_BASE; d=$1
[ -d LogCL ] || git clone -q --depth 1 https://github.com/WeiChen3690/LogCL.git
cd LogCL; [ -d data_zip ] || unzip -oq data.zip -d data_zip; mkdir -p data models result
cp -n ../RE-GCN/src/hyperparameter_range.py src/ 2>/dev/null
cp data_zip/data/get_his_subg.py data/get_his_subg.py
sed -i "s/^dataset_list = \[\"ICEWS14\", \"ICEWS18\",\"ICEWS05-15\"\]/dataset_list = [\"$d\"]/" data/get_his_subg.py
mkdir -p data/$d; for f in train.txt valid.txt test.txt stat.txt entity2id.txt relation2id.txt; do cp $FINCHRONO_OUT/tkg/$d/$f data/$d/; done
grep -q "'$d'" rgcn/utils.py || sed -i "s/dataset in \['ICEWS18',/dataset in ['$d', 'ICEWS18',/" rgcn/utils.py
grep -q FC_SEED_PATCH src/main.py || sed -i '0,/^import torch/s//import torch  # FC_SEED_PATCH\nimport os as _o, random as _rnd, numpy as _np\n_s = int(_o.environ.get("SEED", 0)); torch.manual_seed(_s); torch.cuda.manual_seed_all(_s); _np.random.seed(_s); _rnd.seed(_s)/' src/main.py
grep -q FC_SINGLE src/main.py || python $SCRIPT_DIR/patch_logcl_single.py
cd data; python get_his_subg.py 2>&1 | grep -v -E 'it/s|s/it' | tail -2; du -sh $d; echo LC_DATA_OK

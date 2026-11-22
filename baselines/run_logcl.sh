#!/bin/bash
# run_logcl.sh <data> <seed> <hist> <epochs>: official ICEWS14 command plus --use-cl (the README command omits the flag but the method needs it), no static graph; model chosen on validation by the official loop
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
export SEED=$2; d=$1; h=$3; ep=${4:-30}; cd $FINCHRONO_BASE/LogCL
L=$FINCHRONO_OUT/baselines/logs/logcl_${d}_h${h}_s$2.log; mkdir -p $(dirname $L)
grep -q FINAL_MARK $L 2>/dev/null && { echo already; exit 0; }
python -u src/main.py -d $d --train-history-len $h --test-history-len $h --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --temperature 0.03 --use-cl --n-epochs $ep 2>&1 | grep --line-buffered -v -E 'it/s|s/it' >> $L
echo FINAL_MARK >> $L

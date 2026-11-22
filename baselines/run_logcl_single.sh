#!/bin/bash
# run_logcl_single.sh <data> <hist> <K> <val|test>: per-query scoring of the trained model (K=-1: all queries)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
cd $FINCHRONO_BASE/LogCL; d=$1; h=$2; K=$3; sp=$4
M=$(ls -t models/$d-len$h-*True*.pt | head -1); [ "$sp" = val ] && V=1 || V=
FC_SINGLE=$K FC_VAL=$V FC_MODEL=$M SEED=0 python -u src/main.py -d $d --train-history-len $h --test-history-len $h --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --temperature 0.03 --use-cl --test 2>&1 | grep -E "FC_SINGLE|all_filter|Traceback|Error" | cut -c1-260

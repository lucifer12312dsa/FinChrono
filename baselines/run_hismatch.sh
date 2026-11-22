#!/bin/bash
# run_hismatch.sh <data> <seed> <his> <epochs>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
export SEED=$2; d=$1; h=${3:-5}; ep=${4:-20}
mkdir -p $FINCHRONO_OUT/baselines/logs; L=$FINCHRONO_OUT/baselines/logs/hismatch_${d}_h${h}_s$2.log
grep -q FINAL_MARK $L 2>/dev/null && { echo already; exit 0; }
cd $FINCHRONO_BASE/HiSMatch/hismatch_preprocess
python get_repetitive_history.py -d $d -his $h > /dev/null || { echo PRE1_FAIL; exit 1; }
python get_history_dict.py -d $d -his $h -start-time 0 -gap 1 > /dev/null || { echo PRE2_FAIL; exit 1; }
cd ../src
python -u main.py -d $d --n-hidden 128 --self-loop --background-len 4 --encoder uvrgcn-sub --bg-encoder uvrgcn-sub --start-time 0 --time-gap 1 --history-len $h --gpu 0 --test 0 --n-epochs $ep 2>&1 | grep -v "Epoch\b.*Loss\b" > $L
python -u main.py -d $d --n-hidden 128 --self-loop --background-len 4 --encoder uvrgcn-sub --bg-encoder uvrgcn-sub --start-time 0 --time-gap 1 --history-len $h --gpu 0 --test 1 2>&1 | tail -30 >> $L
grep -q 'Traceback' $L || echo FINAL_MARK >> $L

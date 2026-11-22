#!/bin/bash
# run_tirgn.sh <data> <seed> <hist> <epochs> [hrate]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
export SEED=$2; d=$1; h=$3; ep=${4:-30}; hr=${5:-0.3}
mkdir -p $FINCHRONO_OUT/baselines/logs; L=$FINCHRONO_OUT/baselines/logs/tirgn_${d}_h${h}_s$2.log
grep -q "Hits (filter_ent) @ 10" $L 2>/dev/null && [ "$(grep -c '@ 10' $L)" -gt 0 ] && grep -q FINAL_MARK $L && { echo already; exit 0; }
cd $FINCHRONO_BASE/TiRGN/data/$d 2>/dev/null || { echo nodata; exit 1; }
if [ ! -d history ] && [ ! -f history_seq/.ok ]; then cd $FINCHRONO_BASE/TiRGN/src; python get_history.py --dataset $d; touch ../data/$d/history_seq_ok; fi
cd $FINCHRONO_BASE/TiRGN/src
python -u main.py -d $d --history-rate $hr --train-history-len $h --test-history-len $h --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5 --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --n-epochs $ep --grad-norm 1.0 --dropout 0.2 --input-dropout 0.2 --hidden-dropout 0.2 --feat-dropout 0.2 2>&1 | grep -v "Epoch\b.*Loss\b" > $L
echo FINAL_MARK >> $L

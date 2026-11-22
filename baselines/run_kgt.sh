#!/bin/bash
# run_kgt.sh <data> <seed> [rnn_truncate]: official train_DKG_run.py settings (dim 200, KGT 2 layers, 8 heads, lr 5e-4, up to 150 epochs, patience 10) on our splits
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
d=$1; seed=${2:-0}; cd $FINCHRONO_BASE/FinDKG
L=$FINCHRONO_OUT/baselines/logs/kgt_${d}_s$seed.log; mkdir -p $(dirname $L) result/$d/DKG
sed "s/^graph_mode = .*/graph_mode = \"$d\"/; s/^random_seed = .*/random_seed = $seed/; s/^args.seed = random_seed.*/args.seed = random_seed; args.gpu = 0; args.rnn_truncate_every = ${3:-100}/" train_DKG_run.py > run_$d.py
cp $SCRIPT_DIR/fc_compat.py .; sed -i "1i import fc_compat" run_$d.py
python -u run_$d.py 2>&1 | grep --line-buffered -v -E '^\s*$|batch-[0-9]+\]:' | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E 'it/s|s/it' >> $L
echo FINAL_MARK >> $L

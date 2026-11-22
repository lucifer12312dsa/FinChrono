#!/bin/bash
# run_tlogic_apply.sh <data> <window> <procs>: rules exist already; apply on test and evaluate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
d=$1; w=$2; p=$3; L=$FINCHRONO_OUT/baselines/logs/tlogic_${d}_s0.log
cd $FINCHRONO_BASE/TLogic/mycode; RF=$(ls -t ../output/$d/*rules.json | grep -v cands | head -1 | xargs -I{} basename "{}")
rm -f ../output/$d/*cands*.json
python -u apply.py -d $d -r "$RF" -l 1 2 3 -w $w -p $p --test_data test >> $L 2>&1
ls -t ../output/$d/*cands*.json | head -1 | while read f; do echo "EVAL $f" >> $L; python -u evaluate.py -d $d -c "$(basename "$f")" --test_data test >> $L 2>&1; done
echo FINAL_MARK >> $L

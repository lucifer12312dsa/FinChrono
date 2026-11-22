#!/bin/bash
# run_tlogic.sh <data> <seed> [workers]: official TLogic (CPU). Converts folder to TLogic's text format with json maps.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
d=$1; seed=$2; p=${3:-2}; w=${4:-0}; L=$FINCHRONO_OUT/baselines/logs/tlogic_${d}_s$seed.log; mkdir -p $(dirname $L)
grep -q FINAL_MARK $L 2>/dev/null && { echo already; exit 0; }
python - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]; src = f"{os.environ['FINCHRONO_OUT']}/tkg/{d}"; dst = f"{os.environ['FINCHRONO_BASE']}/TLogic/data/{d}"; os.makedirs(dst, exist_ok=True)
ent = [l.rstrip('\n').split('\t')[0] for l in open(f'{src}/entity2id.txt')]; rel = [l.rstrip('\n').split('\t')[0] for l in open(f'{src}/relation2id.txt')]
safe = lambda s, i: f'{i}_' + s.replace(' ', '_').replace('\t', '_')
E = {safe(x, i): i for i, x in enumerate(ent)}; R = {safe(x, i): i for i, x in enumerate(rel)}
json.dump(E, open(f'{dst}/entity2id.json', 'w')); json.dump(R, open(f'{dst}/relation2id.json', 'w'))
inv_e = {v: k for k, v in E.items()}; inv_r = {v: k for k, v in R.items()}; ts = set()
for n in ('train', 'valid', 'test'):
    with open(f'{dst}/{n}.txt', 'w') as f:
        for l in open(f'{src}/{n}.txt'):
            s, r, o, t = l.split('\t')[:4]; ts.add(int(t)); f.write(f'{inv_e[int(s)]}\t{inv_r[int(r)]}\t{inv_e[int(o)]}\t{t}\n')
json.dump({str(t): i for i, t in enumerate(sorted(ts))}, open(f'{dst}/ts2id.json', 'w'))
PY
cd $FINCHRONO_BASE/TLogic/mycode
uv pip install -q tqdm 2>/dev/null
RUN="python -u"
# learn.py divides the relations among workers, so more workers than relations yields no rules
PL=$(( p < 12 ? p : 12 ))
$RUN learn.py -d $d -l 1 2 3 -n 200 -p $PL -s $seed > $L 2>&1
# TLogic has no hyperparameter chosen here, so only the test split is scored (the candidate file name does not encode the split)
RF=$(ls -t ../output/$d/*rules.json 2>/dev/null | grep -v cands | head -1 | xargs -I{} basename "{}")
echo RULES $RF >> $L
$RUN apply.py -d $d -r "$RF" -l 1 2 3 -w $w -p $p --test_data test >> $L 2>&1
ls -t ../output/$d/*cands*.json | head -2 | while read f; do echo "EVAL $f" >> $L; $RUN evaluate.py -d $d -c "$(basename "$f")" --test_data $(echo "$f" | grep -q valid && echo valid || echo test) >> $L 2>&1; done
echo FINAL_MARK >> $L

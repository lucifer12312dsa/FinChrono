#!/bin/bash
# KGTransformer (official FinDKG repo) on our splits. Adds inverse relations so it answers subject and object queries like the other methods,
# and replaces its raw ranks by the same time-aware filtered ranks. Requires a PyTorch + DGL environment (optionally via TKG_ENV_ACTIVATE).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINCHRONO_OUT="${FINCHRONO_OUT:-$REPO_ROOT/out}"
FINCHRONO_BASE="${FINCHRONO_BASE:-$REPO_ROOT/external_baselines}"
mkdir -p "$FINCHRONO_BASE" "$FINCHRONO_OUT/baselines/logs"
[ -n "${TKG_ENV_ACTIVATE:-}" ] && . "$TKG_ENV_ACTIVATE"
export FINCHRONO_OUT FINCHRONO_BASE
set -e
cd $FINCHRONO_BASE; [ -d FinDKG ] || git clone -q --depth 1 https://github.com/xiaohui-victor-li/FinDKG.git
cd FinDKG; uv pip install -q torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html 2>&1 | tail -1
python - <<'PY'
import numpy as np, os, re
ROOT=os.environ['FINCHRONO_OUT'] + '/tkg'; src=os.environ['FINCHRONO_BASE'] + '/FinDKG/FinDKG_dataset/FinDKG-full'
for d in ('findkg','gdeltfin'):
    out=f"{os.environ['FINCHRONO_BASE']}/FinDKG/data/{d}"; os.makedirs(out,exist_ok=True)
    N,R=[int(x) for x in open(f'{ROOT}/{d}/stat.txt').read().split()[:2]]
    for n in ('train','valid','test'):
        q=np.loadtxt(f'{ROOT}/{d}/{n}.txt',dtype=np.int64,usecols=(0,1,2,3)).reshape(-1,4)
        inv=q[:,[2,1,0,3]].copy(); inv[:,1]+=R; a=np.concatenate([q,inv]); a=a[np.argsort(a[:,3],kind='stable')]
        with open(f'{out}/{n}.txt','w') as f:
            for s,r,o,t in a: f.write(f'{s}\t{r}\t{o}\t{t}\t0\n')
    open(f'{out}/stat.txt','w').write(f'{N}\t{2*R}\t0\n')
    if d=='findkg':
        L=[l.rstrip('\n').split('\t') for l in open(f'{src}/entity2id.txt')]
        assert len(L)==N and all(int(x[1])==i for i,x in enumerate(L)), 'entity ids differ from source'
        open(f'{out}/entity2id.txt','w').write('\n'.join('\t'.join(x[:4]) for x in L)+'\n')
    print(d,N,R,flush=True)
p='DKG/settings.py'; s=open(p).read()
if 'FC_PATCH' not in s:
    s=s.replace('DGL_GRAPH_ID_TYPE =',"ALL_GRAPHS = ALL_GRAPHS + ['findkg', 'gdeltfin']  # FC_PATCH\nDGL_GRAPH_ID_TYPE =",1); open(p,'w').write(s)
p='DKG/data.py'; s=open(p).read()
s=s.replace("if graph_name in ['FinDKG', 'FinDKG-live']:","if graph_name in ['FinDKG', 'FinDKG-live', 'findkg']:"); open(p,'w').write(s)
p='DKG/eval.py'; s=open(p).read()
if 'FC_FILTER' not in s:
    s=s.replace("def eval_link_prediction(model:", '''_FC = {}
def _fc_others(G, t, s, r, o):  # FC_FILTER: other true objects of (s, r) at time t
    key = G.name
    if key not in _FC:
        d = {}; u, v = G.edges()
        for a, b, c, x in zip(u.tolist(), G.edata['rel_type'].tolist(), v.tolist(), G.edata['time'].tolist()):
            d.setdefault((x, a, b), []).append(c)
        _FC[key] = d
    return [x for x in _FC[key].get((t, s, r), []) if x != o]

def eval_link_prediction(model:''',1)
    a="                    pred = target_entity_weight * target_entity_log_prob + time_weight * time_log_prob\n"
    assert s.count(a)>=1
    s=s.replace(a,a+'''                    assert float(eval_times[0]) == float(eval_times[-1])
                    _oth = _fc_others(entire_G, float(eval_times[0]), int(s), int(r), int(o))
                    if _oth:
                        pred = pred.clone(); pred[_oth] = -1e20
''',1)
    open(p,'w').write(s)
print('patched')
PY
echo KGT_SETUP_OK

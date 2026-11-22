#!/usr/bin/env python3
"""Build TKG benchmark folders (RE-GCN file format: s r o t 0) with a 70/15/15 chronological split on the time window.
  prep_tkg.py findkg   -> out/tkg/findkg   FinDKG-full, weekly snapshots
  prep_tkg.py gdelt    -> out/tkg/gdeltfin GDELT MNC events, daily snapshots
"""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
_TMP_DIR.mkdir(parents=True, exist_ok=True)
import collections, gzip, json, os, subprocess, sys
import numpy as np
ROOT = str(_OUT_DIR / 'tkg')

def write(name, quads, ent_names, rel_names, meta):
    d = f'{ROOT}/{name}'; os.makedirs(d, exist_ok=True)
    q = np.array(sorted(set(quads), key=lambda x: (x[3], x[0], x[1], x[2])), dtype=np.int64)
    t0, t1 = q[:, 3].min(), q[:, 3].max()
    a, b = t0 + 0.70 * (t1 - t0), t0 + 0.85 * (t1 - t0)
    tr, va, te = q[q[:, 3] <= a], q[(q[:, 3] > a) & (q[:, 3] <= b)], q[q[:, 3] > b]
    for n, x in (('train', tr), ('valid', va), ('test', te)):
        with open(f'{d}/{n}.txt', 'w') as f:
            for s, r, o, t in x:
                f.write(f'{s}\t{r}\t{o}\t{t}\t0\n')
    with open(f'{d}/entity2id.txt', 'w') as f:
        for i, n in enumerate(ent_names):
            f.write(f'{n}\t{i}\n')
    with open(f'{d}/relation2id.txt', 'w') as f:
        for i, n in enumerate(rel_names):
            f.write(f'{n}\t{i}\n')
    with open(f'{d}/stat.txt', 'w') as f:
        f.write(f'{len(ent_names)}\t{len(rel_names)}\t0\n')
    # learned-model side data: fraction of test objects never seen as (s,r,o) before the query
    seen = {(s, r, o) for s, r, o, _ in np.concatenate([tr, va])}
    meta.update(n_ent=len(ent_names), n_rel=len(rel_names), n_quads=len(q), train=len(tr), valid=len(va), test=len(te),
                t_range=[int(t0), int(t1)], split=[float(a), float(b)], snapshots=int(len(np.unique(q[:, 3]))),
                test_unseen_triple_frac=float(np.mean([(s, r, o) not in seen for s, r, o, _ in te])))
    json.dump(meta, open(f'{d}/meta.json', 'w'), indent=1); print(json.dumps(meta), flush=True)

def findkg():
    src = str(_BASE_DIR / 'FinDKG' / 'FinDKG_dataset' / 'FinDKG-full')
    if not os.path.exists(src):
        subprocess.check_call(['git', 'clone', '-q', '--depth', '1', 'https://github.com/xiaohui-victor-li/FinDKG.git', str(_BASE_DIR / 'FinDKG')])
    ents = [l.rstrip('\n').split('\t')[0] for l in open(f'{src}/entity2id.txt')]
    rels = [l.rstrip('\n').split('\t')[0] for l in open(f'{src}/relation2id.txt')]
    quads = []
    for n in ('train', 'valid', 'test'):
        for l in open(f'{src}/{n}.txt'):
            p = l.split('\t')
            quads.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
    print('raw lines', len(quads))
    write('findkg', quads, ents, rels, {'source': 'FinDKG-full', 'raw_lines': len(quads), 'time_unit': 'week'})

GENERIC = {'COMPANY', 'BUSINESS', 'INDUSTRY', 'BANK', 'COMPANIES', 'AIRLINE', 'EMPLOYER', 'INVESTOR', 'PRODUCER', 'CORPORATION', 'BOSS', 'MANUFACTURER',
           'ENTREPRENEUR', 'TRADER', 'BUSINESSMAN', 'FIRM', 'RETAILER', 'SUPPLIER', 'CONTRACTOR', 'OPERATOR', 'BANKER', 'BUSINESSMEN', 'INVESTORS', 'BANKS',
           'EXECUTIVE', 'CHIEF EXECUTIVE', 'CONSUMER', 'CUSTOMER', 'CITIZEN', 'RESIDENT', 'PEOPLE', 'WORKER', 'EMPLOYEE', 'COMMUNITY', 'SCHOOL', 'UNIVERSITY',
           'STUDENT', 'POLICE', 'COURT', 'JUDGE', 'LAWYER', 'ATTORNEY', 'PRESIDENT', 'MEDIA', 'NEWSPAPER', 'JOURNALIST', 'AUTHORITIES', 'GOVERNMENT', 'MINISTRY', 'THE US', 'CRIMINAL', 'THIEF', 'HOSPITAL', 'FARMER', 'PRISON', 'MILITARY'}

def gdelt(target=320000, min_ent_events=20):
    import datetime as dt
    rows = []
    with gzip.open(str(_OUT_DIR / 'raw' / 'gdelt' / 'events.tsv.gz'), 'rt') as f:
        next(f)
        for l in f:
            p = l.rstrip('\n').split('\t')
            if p[1] in GENERIC or p[4] in GENERIC or p[1] == p[4]:
                continue
            rows.append((p[0], p[1], p[4], p[7], int(p[10] or 0), int(p[11] or 0), p[2], p[5], float(p[9] or 0), int(p[8] or 0)))
    print('named rows', len(rows), flush=True)
    # daily dedup: mentions are summed per (day, s, code, o)
    agg = collections.defaultdict(lambda: [0, 0]); side = {}
    for d, a, b, c, m, s_, t1, t2, g, qc in rows:
        k = (d, a, c, b); agg[k][0] += m; agg[k][1] += s_; side[k] = (g, qc)
    print('daily unique', len(agg), flush=True)
    # corroboration threshold chosen to land near the target size (keeps well-reported events)
    ment = np.array([v[0] for v in agg.values()])
    print('mention quantiles', np.percentile(ment, [10, 25, 50, 75, 90, 95]).tolist(), flush=True)
    for thr in range(1, 200):
        keep = {k: v for k, v in agg.items() if v[0] >= thr}
        cnt = collections.Counter()
        for (d, a, c, b) in keep:
            cnt[a] += 1; cnt[b] += 1
        ents = {e for e, n in cnt.items() if n >= min_ent_events}
        n = sum(1 for (d, a, c, b) in keep if a in ents and b in ents)
        print('thr', thr, 'events', n, 'ents', len(ents), flush=True)
        if n <= target:
            break
    ents = sorted(ents); eid = {e: i for i, e in enumerate(ents)}
    rels = sorted({k[2] for k in keep}); rid = {r: i for i, r in enumerate(rels)}
    d0 = dt.date(2018, 1, 1); quads = []; risk = {}
    for (d, a, c, b) in keep:
        if a in eid and b in eid:
            t = (dt.date(int(d[:4]), int(d[4:6]), int(d[6:])) - d0).days
            quads.append((eid[a], rid[c], eid[b], t))
    gold = collections.defaultdict(list); qcs = {}
    for k, (g, qc) in side.items():
        gold[k[2]].append(g); qcs[k[2]] = qc
    write('gdeltfin', quads, ents, rels, {'source': 'GDELT 1.0 MNC events 2018-2023', 'mention_threshold': thr, 'time_unit': 'day'})
    json.dump({r: {'goldstein': float(np.mean(gold[r])), 'quad': qcs[r]} for r in rels}, open(f'{ROOT}/gdeltfin/rel_side.json', 'w'))

if __name__ == '__main__':
    {'findkg': findkg, 'gdelt': gdelt}[sys.argv[1]]()

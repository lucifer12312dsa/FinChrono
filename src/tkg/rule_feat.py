#!/usr/bin/env python3
"""Temporal path-rule features for the Hawkes encoder (rule excitation channel).
Confidence of a body B for a head relation r is estimated on the training window only: P(o is a true object of (s,r,t) | o is reached from s by B strictly before t).
Bodies: one hop r1, two hops r1,r2. For every query (s,r,t) of train/valid/test, candidates reached through history strictly before t receive
  f1 = noisy-or of body confidences, f2 = max over groundings of a*conf + (1-a)*exp(-lam*dt).
Writes out/rules/<data>/rules.npz (keys sorted, top-K candidates per query)."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='findkg'); ap.add_argument('--K', type=int, default=128); ap.add_argument('--H1', type=int, default=512); ap.add_argument('--Hx', type=int, default=64); ap.add_argument('--H2', type=int, default=64)
ap.add_argument('--n_conf', type=int, default=40000); ap.add_argument('--procs', type=int, default=11); ap.add_argument('--smooth', type=float, default=20.0); ap.add_argument('--a', type=float, default=0.1); ap.add_argument('--lam', type=float, default=0.1)
a = ap.parse_args()
D = f'{_OUT_DIR}/tkg/{a.data}'; OD = f'{_OUT_DIR}/rules/{a.data}'; os.makedirs(OD, exist_ok=True)
rd = lambda n: np.loadtxt(f'{D}/{n}.txt', dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
tr, va, te = rd('train'), rd('valid'), rd('test'); N, R = [int(x) for x in open(f'{D}/stat.txt').read().split()[:2]]; R2 = 2 * R
inv = lambda q: np.concatenate([q, np.stack([q[:, 2], q[:, 1] + R, q[:, 0], q[:, 3]], 1)])
allq = inv(np.concatenate([tr, va, te])); trq = inv(tr)
o_ = np.lexsort((allq[:, 3], allq[:, 0])); E, ER, EO, ET = allq[o_, 0], allq[o_, 1], allq[o_, 2], allq[o_, 3]
TM = int(allq[:, 3].max()) + 2; KEY = E * TM + ET
def before(s, t, H):
    hi = np.searchsorted(KEY, s * TM + t); lo = max(np.searchsorted(KEY, s * TM), hi - H)
    return ER[lo:hi], EO[lo:hi], ET[lo:hi]
def groundings(s, t):
    """yield (body id, object, time of the last edge). body id: r1 for one hop, R2 + r1*R2 + r2 for two hops"""
    r1, o1, t1 = before(s, t, a.H1)
    out = [(int(x), int(y), int(z)) for x, y, z in zip(r1, o1, t1)]
    seen = {}
    for x, y in zip(r1[::-1], o1[::-1]):     # latest relation per neighbour, newest first
        if int(y) not in seen and int(y) != s:
            seen[int(y)] = int(x)
        if len(seen) >= a.Hx:
            break
    for xn, rr in seen.items():
        r2, o2, t2 = before(xn, t, a.H2)
        out += [(R2 + rr * R2 + int(x), int(y), int(z)) for x, y, z in zip(r2, o2, t2) if int(y) != s]
    return out
# ---------- confidence on the training window
t0 = int(tr[:, 3].min() + 0.2 * (tr[:, 3].max() - tr[:, 3].min()))
cand = trq[trq[:, 3] > t0]; truth = {}
for s, r, o, t in cand:
    truth.setdefault((int(s), int(r), int(t)), set()).add(int(o))
keys = list(truth); np.random.RandomState(0).shuffle(keys); keys = keys[:a.n_conf]
def conf_one(k):
    s, r, t = k; g = groundings(s, t); by = {}
    for b, o, _ in g:
        by.setdefault(b, set()).add(o)
    tr_ = truth[k]
    return [(b, r, len(v & tr_), len(v)) for b, v in by.items()]
t1 = time.time()
with Pool(a.procs) as p:
    res = p.map(conf_one, keys, chunksize=200)
hit, siz = {}, {}
for rows in res:
    for b, r, h, n in rows:
        hit[(b, r)] = hit.get((b, r), 0) + h; siz[(b, r)] = siz.get((b, r), 0) + n
CONF = {k: hit[k] / (siz[k] + a.smooth) for k in hit if hit[k] > 0}
print('rules', len(CONF), 'sec', time.time() - t1, 'top', sorted(CONF.values())[-5:], flush=True)
# ---------- features per unique query
Q = inv(np.concatenate([tr, va, te])); uk = np.unique((Q[:, 0] * R2 + Q[:, 1]) * TM + Q[:, 3])
def feat(key):
    t = int(key % TM); c = int(key // TM); s, r = c // R2, c % R2
    f1, f2 = {}, {}; used = set()
    for b, o, te_ in groundings(int(s), t):
        cf = CONF.get((b, int(r)))
        if cf is None:
            continue
        if (b, o) not in used:
            used.add((b, o)); f1[o] = 1 - (1 - f1.get(o, 0.0)) * (1 - min(cf, .999))
        v = a.a * cf + (1 - a.a) * np.exp(-a.lam * (t - te_))
        if v * cf > f2.get(o, 0.0):
            f2[o] = float(v * cf)
    items = sorted(f1.items(), key=lambda x: -x[1])[:a.K]
    co = np.full(a.K, -1, dtype=np.int32); x1 = np.zeros(a.K, dtype=np.float32); x2 = np.zeros(a.K, dtype=np.float32)
    for i, (o, v) in enumerate(items):
        co[i] = o; x1[i] = v; x2[i] = f2.get(o, 0.0)
    return co, x1, x2
t1 = time.time(); C, F1, F2 = [], [], []
with Pool(a.procs) as p:
    for i, (co, x1, x2) in enumerate(p.imap(feat, uk.tolist(), chunksize=200)):
        C.append(co); F1.append(x1); F2.append(x2)
        if i % 20000 == 0:
            print(i, len(uk), time.time() - t1, flush=True)
np.savez(f'{OD}/rules.npz', keys=uk, cand=np.stack(C), f1=np.stack(F1), f2=np.stack(F2), TM=TM, R2=R2)
json.dump({'n_rules': len(CONF), 'args': vars(a)}, open(f'{OD}/meta.json', 'w')); print('DONE', flush=True)

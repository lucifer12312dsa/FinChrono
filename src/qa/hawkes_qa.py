#!/usr/bin/env python3
"""Multivariate Hawkes process on the company disclosure stream of the train window, exact likelihood.
Types: 8-K item kinds, Cumulative metrics, Exclusive relations. For company c with events (r_i, t_i) on [t0, T]:
 log L = sum_i log( mu[r_i] + sum_{j<i} kappa[r_j, r_i] exp(-beta[r_j] (t_i - t_j)) )
         - sum_r mu[r] (T - t0) - sum_j sum_r kappa[r_j, r] / beta[r_j] * (1 - exp(-beta[r_j] (T - t_j)))
Output: out/qa/hawkes_qa.json with half-lives and the excitation matrix used by Snapshot_State to rank records."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import json, math, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
OUT = str(_OUT_DIR / 'qa'); T_TRAIN = '2018-07-18'; dev = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_parquet(f'{OUT}/stream.parquet', columns=['eid', 'rel', 'tau', 'value', 'tx', 'extra'])
df = df[(df.tx <= T_TRAIN) & (df.tx >= '2000-01-01')]
df = df[~((df.tau == 'Exclusive') & df.extra.str.contains('"close"'))]
df['type'] = np.where(df.tau == 'Transient', df.value, df.rel)
types = sorted(df.type.unique()); tid = {t: i for i, t in enumerate(types)}; K = len(types)
df['d'] = (pd.to_datetime(df.tx) - pd.Timestamp('2000-01-01')).dt.days.astype(float)
df['r'] = df.type.map(tid); df = df.drop_duplicates(['eid', 'd', 'r']).sort_values(['eid', 'd'])
TT = float((pd.Timestamp(T_TRAIN) - pd.Timestamp('2000-01-01')).days)
H = 32; rows_r, rows_t, hist_r, hist_dt, hist_m, comp_r, comp_left, spans = [], [], [], [], [], [], [], []
for e, g in df.groupby('eid', sort=False):
    if len(g) < 3:
        continue
    r = g.r.values; t = g.d.values; spans.append(TT - t[0])
    comp_r.append(r); comp_left.append(TT - t)
    for i in range(1, len(g)):
        # last H events strictly before the current day
        k = np.searchsorted(t, t[i], side='left'); lo = max(0, k - H); n = k - lo; hr = np.zeros(H, dtype=np.int64); hd = np.zeros(H, dtype=np.float32); hm = np.zeros(H, dtype=bool)
        hr[:n] = r[lo:k]; hd[:n] = t[i] - t[lo:k]; hm[:n] = True   # strictly earlier: events of one filing do not excite each other
        rows_r.append(r[i]); hist_r.append(hr); hist_dt.append(hd); hist_m.append(hm)
print('types', K, 'entities', len(spans), 'events', len(rows_r), flush=True)
to = lambda x, ty=None: torch.tensor(np.array(x), device=dev, dtype=ty)
Y, HR, HD, HM = to(rows_r), to(hist_r), to(hist_dt, torch.float32), to(hist_m)
CR, CL = to(np.concatenate(comp_r)), to(np.concatenate(comp_left), torch.float32); S = float(np.sum(spans))
mu = torch.nn.Parameter(torch.full((K,), -7.0, device=dev)); kap = torch.nn.Parameter(torch.full((K, K), -6.0, device=dev)); bet = torch.nn.Parameter(torch.full((K,), -3.0, device=dev))
opt = torch.optim.Adam([mu, kap, bet], lr=0.03); idx = np.arange(len(Y)); N = len(Y); bs = 200000
def nll():
    m, k, b = F.softplus(mu), F.softplus(kap), F.softplus(bet)
    perm = torch.randperm(N, device=dev)[:bs]
    lam = m[Y[perm]] + (k[HR[perm], Y[perm].unsqueeze(-1)] * torch.exp(-b[HR[perm]] * HD[perm]) * HM[perm]).sum(1)
    ll = torch.log(lam + 1e-12).sum() * (N / len(perm))
    comp = m.sum() * S + ((k.sum(1) / b)[CR] * (1 - torch.exp(-b[CR] * CL))).sum()
    return (comp - ll) / N
for st in range(1500):
    loss = nll(); opt.zero_grad(); loss.backward(); opt.step()
    if st % 150 == 0:
        print(st, loss.item(), flush=True)
b = F.softplus(bet).detach().cpu().numpy(); k = F.softplus(kap).detach().cpu().numpy(); m = F.softplus(mu).detach().cpu().numpy()
json.dump({'types': types, 'half_life_days': (math.log(2) / b).tolist(), 'beta': b.tolist(), 'kappa': k.tolist(), 'mu': m.tolist(), 'nll': float(loss.item()), 'events': N}, open(f'{OUT}/hawkes_qa.json', 'w'))
o = np.argsort(math.log(2) / b)
print('half-lives', [(types[i], round(math.log(2) / b[i], 1)) for i in o], flush=True)
a = np.dstack(np.unravel_index(np.argsort(-k.ravel())[:12], k.shape))[0]
print('top excitation', [(types[i], types[j], round(float(k[i, j]), 4)) for i, j in a], flush=True); print('DONE', flush=True)

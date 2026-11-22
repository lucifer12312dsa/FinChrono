#!/usr/bin/env python3
"""Multi-hop shock forecasting on GDELT-Fin with Propagate_Hawkes_Shock.
Task: entity s receives a negative event (onset: none in the previous 14 days) at t. Rank the entities reachable within max_hops on the graph visible at t
by whether they receive a new negative event in (t, t+14]. Only events before t are readable; the labels are future events.
Model: activation flows from s along visible edges with weight kappa[r_edge, c_s] exp(-gamma[r_edge] (t - t_edge)) (edges kept: latest per (u, v, r-group)),
       score(v) = sum_h w_h act_h(v) + mu * self(v), self(v) = v's own decayed negative-event count with learnable half-life.
Budget B: B best-first node expansions (one store read per expansion). Training uses full propagation."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import argparse, collections, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='gdeltfin'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='full'); ap.add_argument('--abl', default='')
ap.add_argument('--gate', type=int, default=0, help='rank only entities with no negative event in the last H days: others cannot be a new onset, by the task definition'); ap.add_argument('--hz', type=int, default=0, help='target intensity: 0 two kernels, 1 five learnable kernels, 2 kernels with a small MLP link'); ap.add_argument('--gbdt', type=int, default=0, help='learned baseline: gradient-boosted trees on history features; same queries, candidates and metrics');ap.add_argument('--dev', type=int, default=0); ap.add_argument('--same_type', type=int, default=0, help='label: new negative event of the same CAMEO root as the shock'); ap.add_argument('--W', type=int, default=180); ap.add_argument('--H', type=int, default=14); ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--n_train', type=int, default=6000); ap.add_argument('--n_eval', type=int, default=3000)
a = ap.parse_args(); abl = set(x for x in a.abl.split(',') if x)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
D = f'{_OUT_DIR}/tkg/{a.data}'; OD = f'{_OUT_DIR}/shock/{a.tag}_s{a.seed}'; os.makedirs(OD, exist_ok=True)
rd = lambda n: np.loadtxt(f'{D}/{n}.txt', dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
tr, va, te = rd('train'), rd('valid'), rd('test'); N, R = [int(x) for x in open(f'{D}/stat.txt').read().split()[:2]]
rel = [l.split('\t')[0] for l in open(f'{D}/relation2id.txt')]
side = json.load(open(f'{D}/rel_side.json'))
root = np.array([int(r[:2]) if r[:2].isdigit() else 0 for r in rel]); G = int(root.max()) + 1
neg = np.array([side[r]['quad'] >= 3 and side[r]['goldstein'] <= -2 for r in rel])
allq = np.concatenate([tr, va, te]); allq = allq[np.argsort(allq[:, 3], kind='stable')]
print('N', N, 'R', R, 'groups', G, 'neg rel', neg.sum(), 'neg events', neg[allq[:, 1]].sum(), flush=True)
t_tr, t_va = tr[:, 3].max(), va[:, 3].max()
negev = allq[neg[allq[:, 1]]]
hits = collections.defaultdict(list); hits_c = collections.defaultdict(list)
for s, r, o, t in negev:
    hits[o].append(t); hits_c[(o, int(root[r]))].append(t)     # recipient of a negative event, overall and by event type
for k in hits:
    hits[k] = np.array(sorted(set(hits[k])))
for k in hits_c:
    hits_c[k] = np.array(sorted(set(hits_c[k])))
def had_c(v, c, t0, t1):
    x = hits_c.get((v, c))
    if x is None:
        return False
    i = np.searchsorted(x, t0, side='right'); return i < len(x) and x[i] <= t1
def had(v, t0, t1):
    """negative event for v with t0 < t <= t1"""
    x = hits.get(v)
    if x is None:
        return False
    i = np.searchsorted(x, t0, side='right'); return i < len(x) and x[i] <= t1
# queries
Q = []
for s, r, o, t in negev:
    if not had(o, t - a.H - 1, t - 1):
        Q.append((o, r, t))
Q = sorted(set(Q), key=lambda x: x[2])
sp = lambda lo, hi: [q for q in Q if lo < q[2] <= hi - (a.H)]
Qtr, Qva, Qte = sp(-1, t_tr), sp(t_tr, t_va), sp(t_va, allq[:, 3].max())
print('queries', len(Qtr), len(Qva), len(Qte), flush=True)
random.Random(1).shuffle(Qtr); random.Random(2).shuffle(Qva); random.Random(3).shuffle(Qte)
Qtr, Qva, Qte = Qtr[:a.n_train], Qva[:a.n_eval], Qte[:a.n_eval]
tt = allq[:, 3]
def edges_at(t):
    """visible edges before t inside the window, latest per (u,v,group), both directions"""
    lo, hi = np.searchsorted(tt, t - a.W), np.searchsorted(tt, t)
    e = allq[lo:hi]
    if not len(e):
        return (np.zeros(0, dtype=int),) * 3 + (np.zeros(0),)
    u = np.concatenate([e[:, 0], e[:, 2]]); v = np.concatenate([e[:, 2], e[:, 0]]); g = np.concatenate([root[e[:, 1]], root[e[:, 1]]]); d = np.concatenate([t - e[:, 3], t - e[:, 3]])
    key = (u * N + v) * G + g
    order = np.lexsort((d, key)); key = key[order]; first = np.r_[True, key[1:] != key[:-1]]
    idx = order[first]
    return u[idx], v[idx], g[idx], d[idx].astype(np.float32)
def self_decay(v, t, rate):
    x = hits.get(v)
    return 0.0
class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.kappa = nn.Parameter(torch.full((G, G), -2.0)); s.beta = nn.Parameter(torch.full((G,), -3.0)); s.mu = nn.Parameter(torch.tensor(0.5)); s.self_b = nn.Parameter(torch.tensor(-3.0))
        s.hw = nn.Parameter(torch.tensor([1.0, 0.5, 0.25])); s.b0 = nn.Parameter(torch.tensor(-3.0)); s.scale = nn.Parameter(torch.tensor([2.0, 2.0, 2.0]))
        s.hz_w = nn.Parameter(torch.zeros(6)); s.hz_b = nn.Parameter(torch.tensor([-3.0, -5.0]))
        if a.hz:
            s.hl = nn.Parameter(torch.log(torch.tensor([30.0, 180.0, 60.0, 14.0, 140.0])))     # log half-lives (days): received x2, sent, activity x2
            s.lin = nn.Parameter(torch.zeros(6)); s.link = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1)) if a.hz == 2 else None
    def weights(s, g, d, c):
        k = F.softplus(s.kappa[g, c])
        beta = torch.full((len(g),), 0.05, device=dev) if 'nodecay' in abl else F.softplus(s.beta)[g]
        return k * torch.exp(-beta * d)
    def propagate(s, u, v, g, d, src, c, hops=3):
        A = s.weights(g, d, c)
        act = torch.zeros(N, device=dev); act[src] = 1.0; out = []
        for h in range(hops):
            nx = torch.zeros(N, device=dev).index_add_(0, v, act[u] * A)
            out.append(nx); act = nx / (nx.max().detach() + 1e-6)
        return out
    def self_score(s, sel_t, t):
        return sel_t
def negcount(t, rates, c=None):
    """(len(rates), N) decayed count of negative events received before t, optionally of one event type"""
    lo = np.searchsorted(negev[:, 3], t - 365); hi = np.searchsorted(negev[:, 3], t); e = negev[lo:hi]
    if c is not None:
        e = e[root[e[:, 1]] == c]
    o = torch.tensor(e[:, 2], device=dev); w = torch.tensor(t - e[:, 3], device=dev, dtype=torch.float32)
    c = torch.zeros(N, device=dev).index_add_(0, o, torch.exp(-rates * w)); return c
def sentcount(t, rate):
    lo = np.searchsorted(negev[:, 3], t - 365); hi = np.searchsorted(negev[:, 3], t); e = negev[lo:hi]
    return torch.zeros(N, device=dev).index_add_(0, torch.tensor(e[:, 0], device=dev), torch.exp(-rate * torch.tensor(t - e[:, 3], device=dev, dtype=torch.float32)))
def quiet(t):
    """entities with no negative event received in the last H days, read strictly before t"""
    lo, hi = np.searchsorted(negev[:, 3], t - a.H), np.searchsorted(negev[:, 3], t)
    q = np.ones(N, dtype=bool); q[negev[lo:hi, 2]] = False; return q
def degree(t, u, v, d):
    c = torch.zeros(N, device=dev); return c.index_add_(0, torch.tensor(u, device=dev), torch.exp(-0.02 * torch.tensor(d, device=dev)))
def reachable(u, v, src, hops=3):
    adj = collections.defaultdict(set)
    for x, y in zip(u.tolist(), v.tolist()):
        adj[x].add(y)
    dist = {src: 0}; fr = [src]
    for h in range(1, hops + 1):
        nf = []
        for x in fr:
            for y in adj[x]:
                if y not in dist:
                    dist[y] = h; nf.append(y)
        fr = nf
    dist.pop(src); return dist
def batch(q):
    s_, r_, t = q; u, v, g, d = edges_at(t); dist = reachable(u, v, s_)
    cand = np.array(sorted(dist))
    if a.same_type:
        c0 = int(root[r_]); y = np.array([had_c(c, c0, t, t + a.H) and not had_c(c, c0, t - a.H - 1, t) for c in cand], dtype=np.float32)
    else:
        y = np.array([had(c, t, t + a.H) and not had(c, t - a.H - 1, t) for c in cand], dtype=np.float32)
    return (torch.tensor(u, device=dev), torch.tensor(v, device=dev), torch.tensor(g, device=dev), torch.tensor(d, device=dev)), cand, y, dist
m = Model().to(dev); opt = torch.optim.Adam([dict(params=[p for n, p in m.named_parameters() if not n.startswith('link')], lr=0.02), dict(params=[p for n, p in m.named_parameters() if n.startswith('link')], lr=0.003)])
def hazard(t, E, c=None):
    """target's own Hawkes baseline: decayed negative events received (two learnable half-lives) and decayed edge activity (two fixed half-lives)"""
    u, v, g, d = E
    if a.hz:
        r = torch.tensor([0.05] * 5, device=dev) if 'nodecay' in abl else math.log(2) / torch.exp(m.hl.clamp(0.0, math.log(730.0)))
        act = lambda x: torch.log1p(torch.zeros(N, device=dev).index_add_(0, v, torch.exp(-x * d)))
        f = torch.stack([torch.log1p(negcount(t, r[0])), torch.log1p(negcount(t, r[1])), torch.log1p(sentcount(t, r[2])), act(r[3]), act(r[4]), torch.log1p(torch.zeros(N, device=dev).index_add_(0, v, torch.ones(len(v), device=dev)))], 1)
        out = f @ m.lin
        return out + m.link(f).squeeze(1) if m.link is not None else out
    r = [F.softplus(m.hz_b[0]), F.softplus(m.hz_b[1])] if 'nodecay' not in abl else [torch.tensor(0.05, device=dev)] * 2
    n1, n2 = torch.log1p(negcount(t, r[0])), torch.log1p(negcount(t, r[1]))
    d1 = torch.log1p(torch.zeros(N, device=dev).index_add_(0, v, torch.exp(-0.05 * d)))
    d2 = torch.log1p(torch.zeros(N, device=dev).index_add_(0, v, torch.exp(-0.005 * d)))
    out = m.hz_w[0] * n1 + m.hz_w[1] * n2 + m.hz_w[2] * d1 + m.hz_w[3] * d2
    if c is not None:
        out = out + m.hz_w[4] * torch.log1p(negcount(t, r[0], c)) + m.hz_w[5] * torch.log1p(negcount(t, r[1], c))
    return out
def score(q, E, cand, hops=3):
    s_, r_, t = q; u, v, g, d = E; c = torch.tensor(int(root[r_]), device=dev)
    act = m.propagate(u, v, g, d, s_, c, hops=hops)
    tot = sum(F.softplus(m.hw[h]) * torch.log1p(F.softplus(m.scale[h]) * act[h]) for h in range(hops))
    if 'nomutual' in abl:      # no cross-hop excitation at all: only the target's own decaying intensity remains
        tot = torch.zeros(N, device=dev) + 0 * tot.sum()
    sc = tot + hazard(t, E, int(root[r_]) if a.same_type else None) + m.b0
    if a.gate:
        sc = sc.masked_fill(~torch.tensor(quiet(t), device=dev), -1e4)
    return sc[torch.tensor(cand, device=dev)]
def ap_score(y, s):
    """tie-aware average precision: tied scores share one threshold"""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.sum() > 0 else float('nan')
def evaluate(Qs, fn, name, ks=(50,), reached_only=False):
    """pooled AUC-PR over all query-candidate pairs (queries with no positive stay in as negatives); Recall@K averaged over queries that have a positive"""
    allY, allS, allH = [], [], []; rec = collections.defaultdict(list); n = 0; npos = 0
    for q in Qs:
        E, cand, y, dist = batch(q)
        if len(cand) == 0:
            continue
        s = fn(q, E, cand); n += 1
        if a.gate:      # same rule for every method: scored last, all tied
            s = np.where(quiet(q[2])[cand], s, s.min() - 1.0)
        hop = np.array([dist[int(c)] for c in cand])
        allY.append(y); allS.append(s); allH.append(hop)
        if y.sum() == 0:
            continue
        npos += 1; o = np.argsort(-s, kind='stable')
        if reached_only:      # a candidate never reached within the budget cannot be predicted
            o = o[s[o] > 0]
        for k in ks:
            rec[f'R@{k}'].append(y[o[:k]].sum() / y.sum())
            for h in (1, 2, 3):
                if y[hop == h].sum() > 0:
                    rec[f'R@{k}_h{h}'].append(y[(hop == h)][np.argsort(-s[hop == h], kind='stable')[:k]].sum() / y[hop == h].sum())
    Y, S, H = np.concatenate(allY), np.concatenate(allS), np.concatenate(allH)
    res = dict(name=name, n=n, n_with_positive=npos, base_rate=float(Y.mean()), auc_pr=ap_score(Y, S), **{k: float(np.mean(v)) for k, v in rec.items()})
    for h in (1, 2, 3):
        res[f'auc_pr_h{h}'] = ap_score(Y[H == h], S[H == h])
    return res
def gbdt_baseline():
    """Learned baseline. Every feature reads events strictly before t; hyper-parameters come from validation; test is scored once with the chosen model."""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import average_precision_score
    names = [f'neg_{w}d' for w in (7, 14, 30, 90, 365)] + ['neg_dec05', 'neg_dec005', 'type_dec05', 'type_dec005', 'sent_90d', 'days_since_neg', 'hop', 'deg', 'act05', 'act005', 'link_s', 'link_s_dec', 'ppr', 'src_neg_90d', 'src_deg', 'root', 'quiet']
    def feats(q, E, cand):
        s_, r_, t = q; u, v, g, d = [x.cpu().numpy() for x in E]; c0 = int(root[r_])
        lo, hi = np.searchsorted(negev[:, 3], t - 365), np.searchsorted(negev[:, 3], t); e = negev[lo:hi]; w = (t - e[:, 3]).astype(np.float64)
        cnt = lambda idx, wt=None: np.bincount(idx, weights=wt, minlength=N).astype(np.float64)
        X = [cnt(e[w <= k, 2]) for k in (7, 14, 30, 90, 365)]
        X += [cnt(e[:, 2], np.exp(-.05 * w)), cnt(e[:, 2], np.exp(-.005 * w))]
        same = root[e[:, 1]] == c0
        X += [cnt(e[same, 2], np.exp(-.05 * w[same])), cnt(e[same, 2], np.exp(-.005 * w[same])), cnt(e[w <= 90, 0])]
        last = np.full(N, 9999.0)     # events are sorted by time, so the last write per entity is its most recent one
        last[e[:, 2]] = t - e[:, 3]
        dist = reachable(u, v, s_); hop = np.zeros(N); hop[list(dist)] = list(dist.values())
        deg = cnt(v); a5 = cnt(v, np.exp(-.05 * d)); a05 = cnt(v, np.exp(-.005 * d))
        m_s = u == s_; ls = cnt(v[m_s]); lsd = cnt(v[m_s], np.exp(-.05 * d[m_s]))
        od = np.maximum(cnt(u), 1.0); p = np.zeros(N); p[s_] = 1.0; x = p.copy()    # personalised PageRank, alpha .15, 20 steps
        for _ in range(20):
            x = .15 * p + .85 * cnt(v, x[u] / od[u])
        X += [last, hop, deg, a5, a05, ls, lsd, x]
        X += [np.full(N, X[3][s_]), np.full(N, deg[s_]), np.full(N, float(c0)), quiet(t).astype(np.float64)]
        return np.stack(X, 1)[cand]
    t0 = time.time(); rng = np.random.RandomState(a.seed); XX, YY = [], []
    for q in Qtr:
        E, cand, y, dist = batch(q)
        if len(cand) == 0:
            continue
        keep = (y > 0) | (rng.rand(len(y)) < .3)    # all positives, 30% of negatives
        XX.append(feats(q, E, cand)[keep]); YY.append(y[keep])
    XX, YY = np.concatenate(XX), np.concatenate(YY)
    print('gbdt train rows', XX.shape, 'pos', YY.mean(), 'sec', time.time() - t0, flush=True)
    Va = []
    for q in Qva[:600]:
        E, cand, y, dist = batch(q)
        if len(cand):
            Va.append((feats(q, E, cand), y, quiet(q[2])[cand] if a.gate else np.ones(len(cand), dtype=bool)))
    Xv, Yv, Mv = np.concatenate([x for x, _, _ in Va]), np.concatenate([y for _, y, _ in Va]), np.concatenate([k for _, _, k in Va])
    pv = lambda mdl, X: np.where(Mv, mdl.predict_proba(X)[:, 1], -1.0)
    grid, best_m, best_ap = [], None, -1
    for lr in (.03, .1):
        for leaves in (15, 31, 63):
            for it in (100, 300):
                mdl = HGB(learning_rate=lr, max_leaf_nodes=leaves, max_iter=it, min_samples_leaf=50, l2_regularization=1.0, early_stopping=False, random_state=a.seed).fit(XX, YY)
                ap_v = float(average_precision_score(Yv, pv(mdl, Xv))); grid.append(dict(lr=lr, leaves=leaves, iters=it, val=ap_v)); print(json.dumps(grid[-1]), time.time() - t0, flush=True)
                if ap_v > best_ap:
                    best_ap, best_m = ap_v, mdl
    print('selected', max(grid, key=lambda z: z['val']), flush=True)
    Xp = Xv.copy(); imp = {}
    for j, nm in enumerate(names):      # permutation importance on validation only
        Xp[:, j] = rng.permutation(Xv[:, j]); imp[nm] = best_ap - float(average_precision_score(Yv, pv(best_m, Xp))); Xp[:, j] = Xv[:, j]
    print('importance', json.dumps(dict(sorted(imp.items(), key=lambda z: -z[1]))), flush=True)
    if a.gbdt == 2:
        return
    res = dict(gbdt=evaluate(Qte, lambda q, E, c: best_m.predict_proba(feats(q, E, c))[:, 1], 'GBDT', ks=(20, 50, 100)), grid=grid, val_best=best_ap, features=names, args=vars(a), sec=time.time() - t0)
    json.dump(res, open(f'{OD}/result.json', 'w'), indent=1); print('gbdt', json.dumps(res['gbdt']), flush=True)
if a.gbdt:
    with torch.no_grad():
        gbdt_baseline()
    raise SystemExit
log = []
best, bad = -1, 0
if not os.path.exists(f'{OD}/best.pt'):
    for ep in range(a.epochs):
        random.shuffle(Qtr); tl = 0; cnt = 0; t0 = time.time()
        for q in Qtr:
            E, cand, y, dist = batch(q)
            if len(cand) == 0 or y.sum() == 0:
                continue
            sc = score(q, E, cand); yt = torch.tensor(y, device=dev)
            # pos-vs-neg softmax per query plus BCE
            lp = F.log_softmax(sc, 0); loss = -(lp * yt).sum() / yt.sum() + F.binary_cross_entropy_with_logits(sc, yt, pos_weight=torch.tensor(3.0, device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); cnt += 1
        with torch.no_grad():
            v = evaluate(Qva[:600], lambda q, E, c: score(q, E, c).cpu().numpy(), 'val')
        log.append(dict(ep=ep, loss=tl / max(cnt, 1), val=v, sec=time.time() - t0)); print(json.dumps(log[-1])[:300], flush=True)
        if v['auc_pr'] > best:
            best, bad = v['auc_pr'], 0; torch.save(m.state_dict(), f'{OD}/best.pt')
        else:
            bad += 1
            if bad >= 3:
                break
m.load_state_dict(torch.load(f'{OD}/best.pt', map_location=dev))
if a.dev:
    with torch.no_grad():
        print('VAL', json.dumps(evaluate(Qva[:600], lambda q, E, c: score(q, E, c).cpu().numpy(), 'val', ks=(20, 50, 100))), flush=True)
    print('half-lives', torch.exp(m.hl).tolist() if a.hz else (math.log(2) / F.softplus(m.hz_b)).tolist(), 'weights', m.lin.tolist() if a.hz else m.hz_w.tolist(), flush=True); raise SystemExit
res = {}
with torch.no_grad():
    res['finchrono'] = evaluate(Qte, lambda q, E, c: score(q, E, c).cpu().numpy(), 'FinChrono', ks=(20, 50, 100))
    if not abl:
        def base_deg(q, E, c):
            u, v, g, d = E; x = torch.zeros(N, device=dev).index_add_(0, v, torch.ones(len(v), device=dev)); return x[torch.tensor(c, device=dev)].cpu().numpy()
        res['degree'] = evaluate(Qte, base_deg, 'Degree', ks=(20, 50, 100))
        def recent_neg(q, E, c):
            return torch.log1p(negcount(q[2], torch.tensor(0.05, device=dev)))[torch.tensor(c, device=dev)].cpu().numpy()
        res['recent_neg'] = evaluate(Qte, recent_neg, 'RecentNeg', ks=(20, 50, 100))
        def ppr(q, E, c, beta=0.0, iters=20, al=.15):
            u, v, g, d = E; w = torch.exp(-beta * d) if beta else torch.ones(len(u), device=dev); deg = torch.zeros(N, device=dev).index_add_(0, u, w) + 1e-9
            p = torch.zeros(N, device=dev); p[q[0]] = 1.0; x = p.clone()
            for _ in range(iters):
                x = al * p + (1 - al) * torch.zeros(N, device=dev).index_add_(0, v, x[u] * w / deg[u])
            return x[torch.tensor(c, device=dev)].cpu().numpy()
        res['ppr'] = evaluate(Qte, lambda q, E, c: ppr(q, E, c), 'PPR', ks=(20, 50, 100))
        res['ppr_decay'] = evaluate(Qte, lambda q, E, c: ppr(q, E, c, beta=0.05), 'DecayPPR', ks=(20, 50, 100))
    # budget curve: best-first expansion of B nodes
    def budget_scores(q, E, cand, B, mode):
        s_, r_, t = q; u, v, g, d = E; c0 = int(root[r_])
        A = m.weights(g, d, torch.tensor(c0, device=dev)) if mode == 'hawkes' else (torch.exp(-0.05 * d) if mode == 'decay' else torch.ones_like(d))
        order = torch.argsort(u); uu, vv, AA = u[order], v[order], A[order]
        starts = torch.searchsorted(uu, torch.arange(N + 1, device=dev))
        act = {s_: 1.0}; scores = collections.defaultdict(float); done = set()
        for _ in range(B):
            fr = [(x, w) for x, w in act.items() if x not in done]
            if not fr:
                break
            x = max(fr, key=lambda z: z[1])[0]; done.add(x); lo, hi = starts[x].item(), starts[x + 1].item()
            if hi <= lo:
                continue
            w = AA[lo:hi]; w = w / (w.sum() + 1e-9); nb = vv[lo:hi].tolist()
            for y, wy in zip(nb, w.tolist()):
                act[y] = act.get(y, 0.0) + act[x] * wy; scores[y] += act[x] * wy
        return np.array([scores.get(int(c), 0.0) for c in cand])
    bud = {}
    for mode, nm in (('hawkes', 'FinChrono'), ('decay', 'DecayBFS'), ('uniform', 'UniformBFS')):
        for B in (1, 2, 3, 4, 5):
            def fn(q, E, c, B=B, mode=mode):
                # every policy ranks the entities it reached with the same target hazard, so the curve compares expansion policies only
                s = budget_scores(q, E, c, B, mode); hz = hazard(q[2], E)[torch.tensor(c, device=dev)].cpu().numpy()
                return np.where(s > 0, np.log1p(s * 10) + hz - hz.min() + 1, 0)
            bud[f'{nm}_B{B}'] = evaluate(Qte[:1000], fn, f'{nm}_B{B}', ks=(50,), reached_only=True)
            print(f'{nm}_B{B}', json.dumps(bud[f'{nm}_B{B}'])[:220], flush=True)
    res['budget'] = bud
res['log'] = log; res['args'] = vars(a)
res['half_life_days'] = (math.log(2) / F.softplus(m.beta)).detach().cpu().tolist()
if a.hz:
    res['target_half_lives'] = torch.exp(m.hl.clamp(0.0, math.log(730.0))).detach().cpu().tolist(); res['target_weights'] = m.lin.detach().cpu().tolist()
torch.save(dict(kappa=F.softplus(m.kappa).detach().cpu(), beta=F.softplus(m.beta).detach().cpu()), f'{OD}/dynamics.pt')
json.dump(res, open(f'{OD}/result.json', 'w'), indent=1)
for k in ('finchrono', 'degree', 'recent_neg', 'ppr', 'ppr_decay'):
    if k in res:
        print(k, json.dumps(res[k]), flush=True)

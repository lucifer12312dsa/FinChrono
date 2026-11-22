#!/usr/bin/env python3
"""CT-Graph Hawkes encoder g_phi for TKG extrapolation.
Intensity of object o for query (s,r,t):
  lambda = Softplus( base(s,r,o) + intrinsic + mutual )
  base      = decoder(H_s(t), z_r, ctx)^T H_o(t); H(t) = continuous-time graph attention over events of the last W time units
  intrinsic = sum_{e_i in Hist(s), r_i=r, o_i=o} mu_r exp(-beta_r dt)   + popularity of o under r with relation-specific half-life mixture
  mutual    = sum_{e_j in Hist(s), r_j!=r, o_j=o} alpha_j kappa[r_j,r] exp(-gamma[r_j,r] dt)                       (1-hop cross relation)
            + sum_{s-r1->x in Hist(s)} sum_{x-r2->o in Hist(x)} alpha_{x} kappa2[r2,r] exp(-gamma2[r2,r] dt2)      (2-hop)
Training: point-process NLL (conditional CE + rate term) + temporal InfoNCE of the true object against stale objects of (s,r).
All history lookups use only events strictly before the query time (snapshot isolation)."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def load(d):
    rd = lambda n: np.loadtxt(f'{d}/{n}.txt', dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
    tr, va, te = rd('train'), rd('valid'), rd('test')
    N, R = [int(x) for x in open(f'{d}/stat.txt').read().split()[:2]]
    return tr, va, te, N, R

def add_inverse(q, R):
    inv = q[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
    return np.concatenate([q, inv])

class History:
    """Per-entity event arrays sorted by time, on the GPU; `before(s,t,H)` returns the last H events of s strictly before t."""
    def __init__(self, q, N, device):
        order = np.lexsort((q[:, 3], q[:, 0])); q = q[order]
        self.T = int(q[:, 3].max()) + 2
        self.key = torch.tensor(q[:, 0] * self.T + q[:, 3], device=device)
        self.r = torch.tensor(q[:, 1], device=device); self.o = torch.tensor(q[:, 2], device=device); self.t = torch.tensor(q[:, 3], device=device)
    def before(self, s, t, H):
        hi = torch.searchsorted(self.key, s * self.T + t.clamp(max=self.T - 1), right=False)
        lo = torch.searchsorted(self.key, s * self.T, right=False)
        off = torch.arange(1, H + 1, device=s.device)
        idx = hi.unsqueeze(-1) - off
        mask = idx >= lo.unsqueeze(-1)
        idx = idx.clamp(min=0)
        return self.r[idx], self.o[idx], self.t[idx], mask

class ChannelHistory:
    """Events of each (s, r) channel sorted by time: the repeat term must not be washed out by other relations of a hub entity."""
    def __init__(self, q, R2, device):
        k = q[:, 0] * R2 + q[:, 1]; order = np.lexsort((q[:, 3], k)); q = q[order]; k = k[order]
        self.R2 = R2; self.T = int(q[:, 3].max()) + 2
        self.key = torch.tensor(k * self.T + q[:, 3], device=device); self.o = torch.tensor(q[:, 2], device=device); self.t = torch.tensor(q[:, 3], device=device)
    def before(self, s, r, t, H):
        ch = s * self.R2 + r
        hi = torch.searchsorted(self.key, ch * self.T + t.clamp(max=self.T - 1)); lo = torch.searchsorted(self.key, ch * self.T)
        idx = hi.unsqueeze(-1) - torch.arange(1, H + 1, device=s.device); mask = idx >= lo.unsqueeze(-1); idx = idx.clamp(min=0)
        return self.o[idx], self.t[idx], mask, (hi - lo)

class Popularity:
    """Decayed counts of the (r,o) channel at fixed half-lives, computed from the events strictly before t (stateless, so snapshots may be shuffled).
    A per-relation softmax over the half-lives makes the decay of this channel learnable."""
    def __init__(self, R2, N, hl, ev, horizon):
        self.N, self.R2, self.ev, self.horizon = N, R2, ev, horizon
        self.rate = torch.tensor([math.log(2) / h for h in hl], device=ev[0].device)
        self.c = None
    def at(self, t):
        tt = self.ev[3]; dev = tt.device
        lo = torch.searchsorted(tt, torch.tensor(t - self.horizon, device=dev)); hi = torch.searchsorted(tt, torch.tensor(t, device=dev))
        r, o, te = self.ev[1][lo:hi], self.ev[2][lo:hi], tt[lo:hi]
        idx = r * self.N + o
        w = torch.exp(-self.rate.unsqueeze(-1) * (t - te).float())     # (b, n)
        c = torch.zeros(len(self.rate), self.R2 * self.N, device=dev)
        c.scatter_add_(1, idx.unsqueeze(0).expand(len(self.rate), -1), w)
        self.c = c.view(len(self.rate), self.R2, self.N)
        return self

class Hawkes(nn.Module):
    def __init__(s, N, R2, d=200, hop_m=32, hop_h=16, H=64, abl=(), fix_beta=0.05, hl=(1, 4, 16, 64, 256), W=4, layers=2, ch=50, opt=None):
        super().__init__()
        s.opt = opt or {}
        s.W = W; s.d = d; s.ch = ch
        s.gnn = nn.ModuleList([nn.ModuleDict({'msg': nn.Linear(2 * d + 16, d), 'att': nn.Linear(2 * d + 16, 1), 'upd': nn.Linear(2 * d, d), 'ln': nn.LayerNorm(d)}) for _ in range(layers)])
        s.gdecay_raw = nn.Parameter(torch.full((R2,), -1.0))
        s.dec_bn = nn.BatchNorm1d(3); s.dec_conv = nn.Conv1d(3, ch, 3, padding=1); s.dec_fc = nn.Linear(ch * d, d); s.dec_ln = nn.LayerNorm(d)
        s.N, s.R2, s.H, s.m, s.h2 = N, R2, H, hop_m, hop_h; s.abl = set(abl); s.fix_beta = fix_beta
        s.ent = nn.Embedding(N, d); s.rel = nn.Embedding(R2, d); s.bias = nn.Parameter(torch.full((N,), float(s.opt.get('b0', -12.0))))
        nn.init.xavier_uniform_(s.ent.weight); nn.init.xavier_uniform_(s.rel.weight)
        s.tfreq = nn.Parameter(torch.tensor([1 / 2 ** i for i in range(8)], dtype=torch.float))
        s.att = nn.Sequential(nn.Linear(3 * d + 16, d), nn.GELU(), nn.Linear(d, 1))
        s.ctx = nn.Sequential(nn.Linear(2 * d + 16, d), nn.GELU(), nn.Linear(d, d))
        s.mu = nn.Parameter(torch.ones(R2) * 1.0); s.beta_raw = nn.Parameter(torch.full((R2,), -2.0))
        s.kappa = nn.Parameter(torch.zeros(R2, R2) + 0.05); s.gamma_raw = nn.Parameter(torch.full((R2, R2), -2.0))
        s.kappa2 = nn.Parameter(torch.zeros(R2, R2) + 0.01); s.gamma2_raw = nn.Parameter(torch.full((R2, R2), -2.0))
        s.pop_w = nn.Parameter(torch.zeros(R2, len(hl))); s.pop_scale = nn.Parameter(torch.ones(R2) * 0.5)
        s.hl = hl; s.chan = None; s.Hc = int(s.opt.get('Hc', 128)); s.nu = nn.Parameter(torch.ones(R2)); s.w1 = nn.Parameter(torch.ones(R2) * 0.5); s.w2 = nn.Parameter(torch.ones(R2) * 0.5)
        s.R2 = R2; s.rule = None; s.wr1 = nn.Parameter(torch.ones(R2) * 0.5); s.wr2 = nn.Parameter(torch.ones(R2) * 0.5)
    def beta(s):
        return torch.full_like(s.beta_raw, s.fix_beta) if 'nodecay' in s.abl else F.softplus(s.beta_raw)
    def gamma(s):
        return torch.full_like(s.gamma_raw, s.fix_beta) if 'nodecay' in s.abl else F.softplus(s.gamma_raw)
    def gamma2(s):
        return torch.full_like(s.gamma2_raw, s.fix_beta) if 'nodecay' in s.abl else F.softplus(s.gamma2_raw)
    def tenc(s, dt):
        x = torch.log1p(dt.float()).unsqueeze(-1) * s.tfreq
        return torch.cat([x.sin(), x.cos()], -1)
    def state(s, t, ev):
        """H(t): node states from events with time in [t-W, t). ev = (u, r, v, te) tensors sorted by time."""
        lo = torch.searchsorted(ev[3], torch.tensor(t - s.W, device=ev[3].device)); hi = torch.searchsorted(ev[3], torch.tensor(t, device=ev[3].device))
        h = s.ent.weight
        if hi <= lo:
            return h
        u, r, v, te = ev[0][lo:hi], ev[1][lo:hi], ev[2][lo:hi], ev[3][lo:hi]
        if s.training:
            keep = torch.rand(len(u), device=u.device) > 0.1
            if keep.any():
                u, r, v, te = u[keep], r[keep], v[keep], te[keep]
        dt = (t - te).float(); enc = s.tenc(dt); zr = s.rel(r)
        for l in s.gnn:
            x = torch.cat([zr, h[v], enc], -1)
            a = l['att'](x).squeeze(-1) - F.softplus(s.gdecay_raw)[r] * dt
            amax = torch.full((s.N,), -1e9, device=a.device, dtype=a.dtype).scatter_reduce(0, u, a.detach(), 'amax')
            w = torch.exp(a - amax[u])
            den = torch.zeros(s.N, device=a.device, dtype=a.dtype).index_add_(0, u, w) + 1e-9
            m = torch.zeros_like(h).index_add_(0, u, (w / den[u]).unsqueeze(-1) * F.gelu(l['msg'](x)))
            h = l['ln'](h + F.dropout(l['upd'](torch.cat([h, m], -1)), 0.2, s.training))
        return h
    def forward(s, qs, qr, qt, hist, pop, parts=False, H_t=None):
        B = len(qs); dev = qs.device
        rj, oj, tj, mj = hist.before(qs, qt, s.H)
        dt = (qt.unsqueeze(-1) - tj).clamp(min=0).float()
        zr = s.rel(qr)
        te = s.tenc(dt)
        a = s.att(torch.cat([zr.unsqueeze(1).expand(-1, s.H, -1), s.rel(rj), s.ent(oj), te], -1)).squeeze(-1)
        gate = torch.sigmoid(a) * mj
        att = F.softmax(a.masked_fill(~mj, -1e4), 1) * mj
        ctx = (att.unsqueeze(-1) * s.ctx(torch.cat([s.rel(rj), s.ent(oj), te], -1))).sum(1)
        if s.opt.get('noctx'):
            ctx = ctx * 0
        Hs = H_t if H_t is not None else s.ent.weight
        x = s.dec_bn(torch.stack([Hs[qs], zr, ctx], 1))
        x = F.dropout(F.gelu(s.dec_conv(x)), 0.2, s.training).view(B, -1)
        h = s.dec_ln(s.dec_fc(x))
        if s.opt.get('norm'):
            base = F.normalize(h, dim=-1) @ F.normalize(Hs, dim=-1).t() * s.opt.get('temp', 15.0) + s.opt.get('b0', -12.0)
        else:
            base = h @ Hs.t() + (0 if s.opt.get('nobias') else s.bias)
        same = (rj == qr.unsqueeze(-1))
        rows = torch.arange(B, device=dev).unsqueeze(-1).expand_as(oj)
        if s.chan is not None:
            co, ct, cm, _ = s.chan.before(qs, qr, qt, s.Hc)
            cdt = (qt.unsqueeze(-1) - ct).clamp(min=0).float()
            crow = torch.arange(B, device=dev).unsqueeze(-1).expand_as(co)
            ex = torch.exp(-s.beta()[qr].unsqueeze(-1) * cdt) * cm
            fast = torch.zeros_like(base).index_put((crow[cm], co[cm]), ex[cm], accumulate=True)
            cnt = torch.zeros_like(base).index_put((crow[cm], co[cm]), cm[cm].float(), accumulate=True)
            score = base + F.softplus(s.mu)[qr].unsqueeze(-1) * torch.log1p(fast) + F.softplus(s.nu)[qr].unsqueeze(-1) * torch.log1p(cnt)
        else:
            intr = F.softplus(s.mu)[qr].unsqueeze(-1) * torch.exp(-s.beta()[qr].unsqueeze(-1) * dt) * same * mj
            score = base + torch.zeros_like(base).index_put((rows[mj], oj[mj]), intr[mj], accumulate=True)
        # popularity of o under r (self-excitation of the (r,o) channel)
        w = F.softmax(s.pop_w[qr], -1)
        popv = (w.unsqueeze(-1) * torch.log1p(pop.c[:, qr].permute(1, 0, 2))).sum(1) * F.softplus(s.pop_scale)[qr].unsqueeze(-1)
        score = score + popv
        if s.rule is not None and 'nomutual' not in s.abl:
            # rule excitation: path bodies with confidence estimated on the training window, read from history strictly before t
            qk = (qs * s.R2 + qr) * s.rule['TM'] + qt
            ix = torch.searchsorted(s.rule['keys'], qk)
            assert (ix < len(s.rule['keys'])).all() and (s.rule['keys'][ix] == qk).all(), 'query missing from the rule cache'
            co, f1, f2 = s.rule['cand'][ix], s.rule['f1'][ix], s.rule['f2'][ix]
            mk = co >= 0; rr = torch.arange(B, device=dev).unsqueeze(-1).expand_as(co)
            v = F.softplus(s.wr1)[qr].unsqueeze(-1) * torch.log1p(f1 * 20) + F.softplus(s.wr2)[qr].unsqueeze(-1) * torch.log1p(f2 * 20)
            score = score.index_put((rr[mk], co[mk]), v[mk], accumulate=True)
        if 'nomutual' not in s.abl:
            k1 = s.kappa[rj, qr.unsqueeze(-1).expand_as(rj)] * torch.exp(-s.gamma()[rj, qr.unsqueeze(-1).expand_as(rj)] * dt) * gate * (~same)
            if s.opt.get('mx'):
                k1 = F.softplus(s.kappa)[rj, qr.unsqueeze(-1).expand_as(rj)] * torch.exp(-s.gamma()[rj, qr.unsqueeze(-1).expand_as(rj)] * dt) * gate * (~same)
                m1 = torch.zeros_like(base).index_put((rows[mj], oj[mj]), k1[mj], accumulate=True)
                score = score + F.softplus(s.w1)[qr].unsqueeze(-1) * torch.log1p(m1)
            else:
                score = score.index_put((rows[mj], oj[mj]), k1[mj], accumulate=True)
            m = min(s.m, s.H)
            xs = oj[:, :m]; hop_t = qt.unsqueeze(-1).expand(-1, m)
            r2, o2, t2, m2 = hist.before(xs.reshape(-1), hop_t.reshape(-1), s.h2)
            r2 = r2.view(B, m, -1); o2 = o2.view(B, m, -1); t2 = t2.view(B, m, -1); m2 = m2.view(B, m, -1) & mj[:, :m].unsqueeze(-1)
            dt2 = (qt.view(-1, 1, 1) - t2).clamp(min=0).float()
            qr2 = qr.view(-1, 1, 1).expand_as(r2)
            k2 = s.kappa2[r2, qr2] * torch.exp(-s.gamma2()[r2, qr2] * dt2) * gate[:, :m].unsqueeze(-1)
            rows2 = torch.arange(B, device=dev).view(-1, 1, 1).expand_as(o2)
            if s.opt.get('mx'):
                k2 = F.softplus(s.kappa2)[r2, qr2] * torch.exp(-s.gamma2()[r2, qr2] * dt2) * gate[:, :m].unsqueeze(-1)
                m2v = torch.zeros_like(base).index_put((rows2[m2], o2[m2]), k2[m2], accumulate=True)
                score = score + F.softplus(s.w2)[qr].unsqueeze(-1) * torch.log1p(m2v)
            else:
                score = score.index_put((rows2[m2], o2[m2]), k2[m2], accumulate=True)
        # log Softplus, stable in the far-negative tail where it equals the raw score
        if s.opt.get('raw'):
            loglam = score
        else:
            loglam = torch.where(score < -15, score, torch.log(F.softplus(score.clamp(min=-15))))
        if parts:
            return loglam, rj, oj, tj, mj
        return loglam, rj, oj, mj

def rank_metrics(logl, target, filt):
    """time-aware filtered ranks; filt: bool mask (B,N) of other true objects at this time"""
    logl = logl.masked_fill(filt, -1e9)
    tgt = logl.gather(1, target.unsqueeze(1))
    return (logl > tgt).sum(1).float() + 0.5 * ((logl == tgt).sum(1).float() - 1) + 1

@torch.no_grad()
def evaluate(model, quads, full_hist, pop, N, R, bs, dev, ev):
    """single-step extrapolation with ground-truth history before the query time (standard TKG protocol)"""
    model.eval(); ranks = []
    q2 = add_inverse(quads, R)
    times = np.unique(q2[:, 3])
    for t in times:
        cur = q2[q2[:, 3] == t]
        pop.at(int(t))
        H_t = model.state(int(t), ev)
        kt = {}
        for s_, r_, o_, _ in cur:
            kt.setdefault((s_, r_), []).append(o_)
        for i in range(0, len(cur), bs):
            b = cur[i:i + bs]
            qs = torch.tensor(b[:, 0], device=dev); qr = torch.tensor(b[:, 1], device=dev); qo = torch.tensor(b[:, 2], device=dev); qt = torch.tensor(b[:, 3], device=dev)
            loglam, *_ = model(qs, qr, qt, full_hist, pop, H_t=H_t)
            filt = torch.zeros_like(loglam, dtype=torch.bool)
            idx = [(j, o2) for j, (s_, r_, o_, _) in enumerate(b) for o2 in kt[(s_, r_)] if o2 != o_]
            if idx:
                ij = torch.tensor(idx, device=dev); filt[ij[:, 0], ij[:, 1]] = True
            ranks.append(rank_metrics(loglam, qo, filt).cpu())
    r = torch.cat(ranks).float()
    return {'mrr': (1 / r).mean().item(), 'h1': (r <= 1).float().mean().item(), 'h3': (r <= 3).float().mean().item(), 'h10': (r <= 10).float().mean().item(), 'n': len(r)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--tag'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--bs', type=int, default=4096); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d', type=int, default=200); ap.add_argument('--H', type=int, default=64); ap.add_argument('--m', type=int, default=32); ap.add_argument('--h2', type=int, default=16)
    ap.add_argument('--abl', default=''); ap.add_argument('--rho_pp', type=float, default=0.01); ap.add_argument('--rho_nce', type=float, default=0.5)
    ap.add_argument('--rules', type=int, default=0); ap.add_argument('--W', type=int, default=4); ap.add_argument('--opt', default='{}'); ap.add_argument('--shuffle', type=int, default=1); ap.add_argument('--horizon', type=int, default=512); ap.add_argument('--patience', type=int, default=3); ap.add_argument('--wd', type=float, default=1e-6)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = f'{_OUT_DIR}/tkg/{a.data}'; out = f'{_OUT_DIR}/hawkes/{a.data}/{a.tag}_s{a.seed}'; os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/result.json'):
        print('done', out); return
    tr, va, te, N, R = load(d)
    allq = add_inverse(np.concatenate([tr, va, te]), R)
    hist = History(allq, N, dev)   # lookups are strictly < t, so future events are never visible
    chan = ChannelHistory(allq, 2 * R, dev)
    model = Hawkes(N, 2 * R, d=a.d, H=a.H, hop_m=a.m, hop_h=a.h2, abl=[x for x in a.abl.split(',') if x], W=a.W, opt=json.loads(a.opt)).to(dev)
    if not json.loads(a.opt).get('nochan'):
        model.chan = chan
    if a.rules:
        z = np.load(f'{_OUT_DIR}/rules/{a.data}/rules.npz')
        model.rule = dict(keys=torch.tensor(z['keys'], device=dev), TM=int(z['TM']), cand=torch.tensor(z['cand'].astype(np.int64), device=dev), f1=torch.tensor(z['f1'], device=dev), f2=torch.tensor(z['f2'], device=dev))
    _ord = np.argsort(allq[:, 3], kind='stable'); _e = allq[_ord]
    ev = tuple(torch.tensor(_e[:, i], device=dev) for i in (0, 1, 2, 3))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    pop = Popularity(2 * R, N, model.hl, ev, a.horizon)
    trq = add_inverse(tr, R); times = np.unique(trq[:, 3])
    best, bad = -1, 0; log = []
    start_ep = 0
    if os.path.exists(f'{out}/last.pt'):
        ck = torch.load(f'{out}/last.pt', map_location=dev)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); best, bad, start_ep, log = ck['best'], ck['bad'], ck['ep'] + 1, ck['log']
        print('resumed at epoch', start_ep, flush=True)
    for ep in range(start_ep, a.epochs):
        model.train(); t0 = time.time(); tot = np.zeros(3); steps = 0
        order = np.random.permutation(len(times)) if a.shuffle else np.arange(len(times))
        for ti in order:
            t = times[ti]
            cur = trq[trq[:, 3] == t]; cur = cur[np.random.permutation(len(cur))]
            dtbin = 1.0 if ti == 0 else float(t - times[ti - 1])
            pop.at(int(t))
            for i in range(0, len(cur), a.bs):
                b = cur[i:i + a.bs]
                qs = torch.tensor(b[:, 0], device=dev); qr = torch.tensor(b[:, 1], device=dev); qo = torch.tensor(b[:, 2], device=dev); qt = torch.tensor(b[:, 3], device=dev)
                H_t = model.state(int(t), ev)
                loglam, rj, oj, mj = model(qs, qr, qt, hist, pop, H_t=H_t)
                ce = F.cross_entropy(loglam, qo)
                logtotal = torch.logsumexp(loglam, 1)
                # point-process NLL = CE + (survival over the bin - log total rate); the exponential is linearised above e^3 to keep it bounded
                surv = torch.where(logtotal < 3, logtotal.clamp(max=3).exp(), math.exp(3) * (1 + logtotal - 3))
                pp = (surv * dtbin - logtotal).mean()
                # temporal InfoNCE against stale objects: same (s,r) objects from Hist(s) other than the true object
                stale = (rj == qr.unsqueeze(-1)) & mj & (oj != qo.unsqueeze(-1))
                cand = torch.zeros_like(loglam, dtype=torch.bool)
                rows = torch.arange(len(b), device=dev).unsqueeze(-1).expand_as(oj)
                cand[rows[stale], oj[stale]] = True
                has = cand.any(1)
                nce = torch.tensor(0.0, device=dev)
                if has.any():
                    cand[torch.arange(len(b), device=dev), qo] = True
                    nce = F.cross_entropy(loglam[has].masked_fill(~cand[has], -1e4), qo[has])
                loss = ce + a.rho_pp * pp + a.rho_nce * nce
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += [ce.item(), pp.item(), nce.item()]; steps += 1
        v = evaluate(model, va, hist, pop, N, R, a.bs * 2, dev, ev)
        log.append({'ep': ep, 'loss': (tot / steps).tolist(), 'val': v, 'sec': time.time() - t0}); print(json.dumps(log[-1]), flush=True)
        if v['mrr'] > best:
            best, bad = v['mrr'], 0; torch.save(model.state_dict(), f'{out}/best.pt')
        else:
            bad += 1
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'best': best, 'bad': bad, 'ep': ep, 'log': log}, f'{out}/last.pt')
        if bad >= a.patience:
            break
    model.load_state_dict(torch.load(f'{out}/best.pt', map_location=dev))
    val = evaluate(model, va, hist, pop, N, R, a.bs * 2, dev, ev)
    test = evaluate(model, te, hist, pop, N, R, a.bs * 2, dev, ev)
    rels = [l.split('\t')[0] for l in open(f'{d}/relation2id.txt')]
    rels = rels + [x + '^-1' for x in rels]
    hl = (math.log(2) / model.beta()).detach().cpu().tolist()
    res = {'args': vars(a), 'val': val, 'test': test, 'log': log, 'half_life': dict(zip(rels, hl))}
    torch.save({'kappa': model.kappa.detach().cpu(), 'kappa2': model.kappa2.detach().cpu(), 'gamma': model.gamma().detach().cpu(), 'gamma2': model.gamma2().detach().cpu(), 'mu': model.mu.detach().cpu(), 'beta': model.beta().detach().cpu()}, f'{out}/dynamics.pt')
    json.dump(res, open(f'{out}/result.json', 'w'), indent=1)
    print('TEST', json.dumps(test), flush=True)

if __name__ == '__main__':
    main()

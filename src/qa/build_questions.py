#!/usr/bin/env python3
"""Fin-ChronoQA questions and counterfactual time-flip pairs.
Gold answers come from the bi-temporal timeline of each (entity, relation), reconstructed independently of the store code:
intervals are computed here from open/close events, and only unambiguous anchors are used (exactly one holder, outside coarse-date years).
Split on the anchor time t_q over the window 2000-01-01..2026-06-30: train 70% / val 15% / test 15%.
15% of Wikidata companies are held out of training entirely; their pre-2018 anchors form the `hist` track (historical anchors, future states exist)."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import collections, datetime as dt, json, os, random, re, sys
import numpy as np, pandas as pd
OUT = str(_OUT_DIR / 'qa')
rnd = random.Random(20261105)
D0, D1 = dt.date(2000, 1, 1), dt.date(2026, 6, 30)
span = (D1 - D0).days; T_TRAIN = D0 + dt.timedelta(days=int(.70 * span)); T_VAL = D0 + dt.timedelta(days=int(.85 * span))
print('split', T_TRAIN, T_VAL, flush=True)
df = pd.read_parquet(f'{OUT}/stream.parquet'); ents = pd.read_parquet(f'{OUT}/entities.parquet').set_index('eid')
name = ents.name.to_dict(); cik = ents.cik.to_dict()
dd = dt.date.fromisoformat
def iso(d): return d.isoformat()
def add_days(s, n): return iso(dd(s) + dt.timedelta(days=n))
def fmt_date(s):
    d = dd(s); return rnd.choice([d.strftime('%B %-d, %Y'), s, d.strftime('%b %-d, %Y')])
held = {e for e in ents.index if ents.source[e] == 'wikidata' and rnd.random() < .15}

# ---- Exclusive timelines
ex = df[df.tau == 'Exclusive'].copy(); ex['ex'] = ex.extra.map(json.loads)
TL = {}
for (e, r), g in ex.groupby(['eid', 'rel'], sort=False):
    opens = []
    for x in g.itertuples():
        if x.ex.get('kind') == 'open':
            opens.append(dict(id=x.id, v=x.value, a=x.vfrom, ap=x.vfrom_prec, b=x.vto, bp=x.vto_prec))
    opens.sort(key=lambda z: (z['a'], z['id']))
    closes = {x.value + '|' + x.vfrom: x.id for x in g.itertuples() if x.ex.get('kind') == 'close'}
    # merge repeated values that do not change the holder; a holder's effective end = explicit end or next different holder's start
    for i, z in enumerate(opens):
        nxt = next((w for w in opens[i + 1:] if w['v'] != z['v']), None)
        z['eb'] = min([x for x in [z['b'], nxt['a'] if nxt else None] if x] or ['9999-12-31'])
        z['eb_ap'] = 11 if (z['b'] is None or (nxt and nxt['a'] < z['b'])) and nxt is None else (nxt['ap'] if nxt and (z['b'] is None or nxt['a'] <= z['b']) else (z['bp'] or 11))
    TL[(e, r)] = opens

def holders(tl, t):
    return [z for z in tl if z['a'] <= t and t < z['eb']]
def safe_range(z, tl):
    """anchor interval [a,b) inside which the holder is unambiguous given date precision"""
    a = z['a']; b = z['eb']
    if z['ap'] == 9: a = f"{int(a[:4]) + 1}-01-01"
    elif z['ap'] == 10:
        y, m = int(a[:4]), int(a[5:7]); a = f"{y + (m == 12)}-{1 if m == 12 else m + 1:02d}-01"
    if b != '9999-12-31' and z['eb_ap'] == 9: b = f"{b[:4]}-01-01"
    elif b != '9999-12-31' and z['eb_ap'] == 10: b = f"{b[:7]}-01"
    if b == '9999-12-31':
        b = min(iso(D1), add_days(z['a'], 365 * 12))   # open-ended statements are trusted for 12 years only
    return a, b
def in_split(t):
    if t < iso(D0): return None
    if t <= iso(T_TRAIN): return 'train'
    if t <= iso(T_VAL): return 'val'
    if t <= iso(D1): return 'test'
    return None
def flips_upto(tl, t):
    vals = []
    for z in tl:
        if z['a'] <= t and (not vals or vals[-1] != z['v']): vals.append(z['v'])
    return max(0, len(vals) - 1)
def evidence(tl, z, t):
    return [z['id']]

Q_STATE = {
 'chief_executive': ['Who was the CEO of {e} on {d}?', 'As of {d}, who served as chief executive of {e}?', 'Name the chief executive officer of {e} at {d}.'],
 'chairperson': ['Who chaired the board of {e} on {d}?', 'As of {d}, who was the chairperson of {e}?'],
 'headquarters': ['Where was {e} headquartered on {d}?', 'As of {d}, what was the headquarters location of {e}?'],
 'parent_organization': ['What was the parent organization of {e} on {d}?', 'As of {d}, which group was {e} a subsidiary of?'],
 'owner': ['Who owned {e} on {d}?', 'As of {d}, who was the owner of {e}?'],
 'director_manager': ['Who was the director of {e} on {d}?', 'As of {d}, who managed {e}?'],
 'registered_name': ['What was the SEC registered name of the registrant with CIK {c} on {d}?', 'As of {d}, under which name was CIK {c} registered with the SEC?'],
}
REL_WORD = {'chief_executive': 'CEOs', 'chairperson': 'chairpersons', 'headquarters': 'headquarters locations', 'parent_organization': 'parent organizations', 'owner': 'owners', 'director_manager': 'directors', 'registered_name': 'SEC registered names'}
pool = collections.defaultdict(list)  # (family, split) -> questions
qid = 0
def emit(family, split, track, **kw):
    global qid
    q = dict(qid=f'q{qid}', family=family, split=split, track=track, **kw); qid += 1
    pool[(family, split, track)].append(q)
def ent_ref(e, r):
    return (f'CIK {cik[e].zfill(10)}' if r == 'registered_name' else name[e])
for (e, r), tl in TL.items():
    if e not in name: continue
    for i, z in enumerate(tl):
        a, b = safe_range(z, tl)
        if a >= b: continue
        days = (dd(b) - dd(a)).days
        nxt = next((w for w in tl[i + 1:] if w['v'] != z['v']), None)
        # anchors: uniform inside the interval plus two near the boundaries (hardest for stale / future confusion)
        anchors = [add_days(a, rnd.randrange(days)) for _ in range(3)]
        if days > 40:
            anchors += [add_days(a, rnd.randrange(min(days, 180))), add_days(b, -1 - rnd.randrange(min(days, 180)))]
        for t in sorted(set(anchors)):
            s = in_split(t)
            if not s: continue
            hs = holders(tl, t)
            if len(hs) != 1 or hs[0]['v'] != z['v']: continue
            stale = sorted({w['v'] for w in tl if w['eb'] <= t and w['v'] != z['v']})
            fut = sorted({w['v'] for w in tl if w['a'] > t} - {w['v'] for w in tl if w['a'] <= t})   # first appears after t_q
            f = flips_upto(tl, t); track = 'hist' if e in held else 'main'
            if e in held and s != 'train': track = 'main'
            q = rnd.choice(Q_STATE[r]).format(e=name[e], c=cik.get(e) and cik[e].zfill(10), d=fmt_date(t))
            emit('state', s, track, question=q, eid=e, rel=r, t_q=t, answer=z['v'], stale=stale, future=fut, flips=f, evidence=[z['id']], hops=1, future_flips=len(fut))
# lineage
for (e, r), tl in TL.items():
    if e not in name or len(tl) < 2: continue
    for _ in range(3):
        i = rnd.randrange(len(tl)); j = rnd.randrange(i, len(tl))
        a0, _b = safe_range(tl[i], tl); a1, b1 = safe_range(tl[j], tl)
        if a0 >= _b or a1 >= b1: continue
        t0 = add_days(a0, rnd.randrange(max(1, (dd(_b) - dd(a0)).days))); t1 = add_days(a1, rnd.randrange(max(1, (dd(b1) - dd(a1)).days)))
        if t0 >= t1 or not in_split(t1) or t0 < '1990': continue
        seq = [w for w in tl if w['a'] <= t1 and w['eb'] > t0]
        # strictly sequential only: exactly one holder at every change point and no coarse boundaries inside
        if len(seq) < 2 or any(a['eb'] > b['a'] for a, b in zip(seq, seq[1:])) or any(w['ap'] < 11 and i0 > 0 for i0, w in enumerate(seq)):
            continue
        vals = []
        for w in seq:
            if not vals or vals[-1] != w['v']: vals.append(w['v'])
        if len(vals) < 2: continue
        s = in_split(t1); track = 'hist' if (e in held and s == 'train') else 'main'
        q = f"List in chronological order the {REL_WORD[r]} of {ent_ref(e, r)} between {fmt_date(t0)} and {fmt_date(t1)}."
        emit('lineage', s, track, question=q, eid=e, rel=r, t_q=t1, t_start=t0, answer=' -> '.join(vals), answer_list=vals, stale=[], future=sorted({w['v'] for w in tl if w['a'] > t1} - {w['v'] for w in tl if w['a'] <= t1}), flips=len(vals) - 1, evidence=[w['id'] for w in seq], hops=1)
# two-hop state: CEO/chair of the parent or owner at t
name_rev = {v: k for k, v in name.items()}
for (e, r), tl in TL.items():
    if r not in ('parent_organization', 'owner') or e not in name: continue
    for z in tl:
        p = name_rev.get(z['v'])
        if not p: continue
        for r2 in ('chief_executive', 'chairperson'):
            tl2 = TL.get((p, r2))
            if not tl2: continue
            a, b = safe_range(z, tl)
            for w in tl2:
                a2, b2 = safe_range(w, tl2)
                lo, hi = max(a, a2), min(b, b2)
                if lo >= hi: continue
                for _ in range(2):
                    t = add_days(lo, rnd.randrange((dd(hi) - dd(lo)).days)); s = in_split(t)
                    if not s: continue
                    if len(holders(tl, t)) != 1 or len(holders(tl2, t)) != 1: continue
                    q = f"As of {fmt_date(t)}, who was the {'CEO' if r2 == 'chief_executive' else 'chairperson'} of the {'parent organization' if r == 'parent_organization' else 'owner'} of {name[e]}?"
                    emit('twohop', s, 'hist' if (e in held and s == 'train') else 'main', question=q, eid=e, rel=r, rel2=r2, eid2=p, t_q=t, answer=w['v'], stale=sorted({x['v'] for x in tl2 if x['eb'] <= t and x['v'] != w['v']}), future=sorted({x['v'] for x in tl2 if x['a'] > t} - {x['v'] for x in tl2 if x['a'] <= t}), flips=flips_upto(tl2, t) + flips_upto(tl, t), evidence=[z['id'], w['id']], hops=2)
# ---- Cumulative: latest report known at t (valid time vs filing time), and period-over-period change
cu = df[df.tau == 'Cumulative'].copy(); cu['val'] = cu.extra.map(lambda s: json.loads(s)['val'])
MW = {'revenue': 'annual revenue', 'total_assets': 'total assets', 'total_liabilities': 'total liabilities', 'stockholders_equity': "stockholders' equity", 'net_income': 'annual net income'}
gr = list(cu.groupby(['eid', 'rel'], sort=False))
rnd.shuffle(gr)
for (e, r), g in gr:
    g = g.sort_values(['tx', 'id'])
    recs = g[['id', 'vfrom', 'tx', 'val', 'value']].values.tolist()
    if len(recs) < 3 or e not in name: continue
    for _ in range(2):
        t = add_days(rnd.choice(recs)[2], rnd.randrange(-40, 300)); s = in_split(t)
        if not s or t > iso(D1): continue
        vis = [x for x in recs if x[2] <= t]
        if len(vis) < 2: continue
        per = {}
        for x in vis: per[x[1]] = x
        ends = sorted(per)
        if len(ends) < 2: continue
        cur = per[ends[-1]]
        stale = sorted({per[k][4] for k in ends[:-1]} - {cur[4]})
        fut = sorted({x[4] for x in recs if x[2] > t} - {x[4] for x in vis})
        track = 'main'
        if rnd.random() < .7:
            q = f"As of {fmt_date(t)}, what is the most recently reported {MW[r]} of {name[e]} in USD?"
            emit('metric', s, track, question=q, eid=e, rel=r, t_q=t, answer=cur[4], stale=stale, future=fut, flips=len(ends) - 1, evidence=[cur[0]], hops=1)
        else:
            prev = per[ends[-2]]; delta = cur[3] - prev[3]
            q = f"As of {fmt_date(t)}, by how many USD did the reported {MW[r]} of {name[e]} change between its two most recent fiscal years? Give a signed number."
            emit('delta', s, track, question=q, eid=e, rel=r, t_q=t, answer=f"{int(delta):,}", stale=[], future=[], flips=len(ends) - 1, evidence=[cur[0], prev[0]], hops=1)
# ---- Transient: latest material event known at t
tr = df[df.tau == 'Transient']
gr = list(tr.groupby('eid', sort=False)); rnd.shuffle(gr)
for e, g in gr:
    if e not in name: continue
    recs = g[['id', 'vfrom', 'tx', 'value']].sort_values(['tx', 'id']).values.tolist()
    if len(recs) < 3: continue
    for _ in range(3):
        t = add_days(rnd.choice(recs)[2], rnd.randrange(0, 200)); s = in_split(t)
        if not s or t > iso(D1): continue
        vis = [x for x in recs if x[2] <= t]
        if len(vis) < 2: continue
        last_tx = vis[-1][2]
        lastset = {x[3] for x in vis if x[2] == last_tx}
        if len(lastset) != 1: continue
        cur = vis[-1]
        q = f"As of {fmt_date(t)}, what kind of material event did {name[e]} disclose most recently in an 8-K?"
        emit('event', s, 'main', question=q, eid=e, rel='material_event', t_q=t, answer=cur[3], stale=sorted({x[3] for x in vis[:-1]} - {cur[3]}), future=sorted({x[3] for x in recs if x[2] > t} - {x[3] for x in vis}), flips=0, evidence=[cur[0]], hops=1)
print({k: len(v) for k, v in sorted(pool.items())}, flush=True)
# ---- quotas (6,400 main questions, 70/15/15; state balanced across flip buckets)
FAM = {'state': .50, 'lineage': .12, 'metric': .12, 'delta': .06, 'event': .12, 'twohop': .08}
SPL = {'train': 4480, 'val': 960, 'test': 960}
sel = []
for s, n in SPL.items():
    for fam, p in FAM.items():
        want = int(round(n * p)); cands = pool[(fam, s, 'main')]; rnd.shuffle(cands)
        if fam == 'state':
            # exclude near-duplicate anchors of the same fact; balance flips 0,1,2,3,>=4
            bk = collections.defaultdict(list)
            for q in cands: bk[min(q['flips'], 4)].append(q)
            shares = {0: .15, 1: .30, 2: .25, 3: .15, 4: .15}; got = []
            for b, sh in shares.items():
                got += bk[b][:int(want * sh)]
            rest = [q for q in cands if q not in got][:max(0, want - len(got))] if len(got) < want else []
            got += rest; sel += got[:want]
        else:
            sel += cands[:want]
hist = pool[('state', 'train', 'hist')]; rnd.shuffle(hist)
hist = [q for q in hist if q['future']][:800]
ltr = pool[('lineage', 'train', 'hist')]; thr = pool[('twohop', 'train', 'hist')]; rnd.shuffle(ltr); rnd.shuffle(thr)
sel += hist + ltr[:100] + thr[:100]
json.dump(sel, open(f'{OUT}/questions.json', 'w')); json.dump(sorted(held), open(f'{OUT}/held.json', 'w'))
c = collections.Counter((q['split'], q['track'], q['family']) for q in sel)
print(sorted(c.items()), flush=True)
print('flips(state,test)', collections.Counter(min(q['flips'], 4) for q in sel if q['split'] == 'test' and q['family'] == 'state'), flush=True)
print('with future (test)', sum(bool(q['future']) for q in sel if q['split'] == 'test'), 'with stale', sum(bool(q['stale']) for q in sel if q['split'] == 'test'), flush=True)
# ---- paired time-flip candidates from TRAIN window only (entities outside the held-out set)
pairs = []
for (e, r), tl in TL.items():
    if e in held or e not in name: continue
    for i in range(len(tl) - 1):
        z, w = tl[i], tl[i + 1]
        if z['v'] == w['v'] or z['eb'] > w['a']: continue
        a1, b1 = safe_range(z, tl); a2, b2 = safe_range(w, tl)
        b1 = min(b1, iso(T_TRAIN)); b2 = min(b2, iso(T_TRAIN))
        if a1 >= b1 or a2 >= b2: continue
        pairs.append(dict(eid=e, rel=r, ref=ent_ref(e, r), old=z['v'], new=w['v'], a1=a1, b1=b1, a2=a2, b2=b2, flip=w['a'], old_id=z['id'], new_id=w['id']))
for (e, r), g in cu.groupby(['eid', 'rel'], sort=False):
    if e in held or e not in name: continue
    g = g[g.tx <= iso(T_TRAIN)].sort_values(['tx', 'id']); recs = g[['id', 'vfrom', 'tx', 'value']].values.tolist()
    for x, y in zip(recs, recs[1:]):
        if x[3] == y[3] or y[1] <= x[1]: continue
        b2 = min(add_days(y[2], 200), iso(T_TRAIN))
        if y[2] < b2:
            pairs.append(dict(eid=e, rel=r, ref=name[e], old=x[3], new=y[3], a1=x[2], b1=y[2], a2=y[2], b2=b2, flip=y[2], old_id=x[0], new_id=y[0], cumulative=True))
json.dump(pairs, open(f'{OUT}/flip_pairs.json', 'w'))
print('flip pair sources', len(pairs), collections.Counter(p['rel'] for p in pairs).most_common(12), flush=True)
print('DONE', flush=True)

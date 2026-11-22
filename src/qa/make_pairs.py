#!/usr/bin/env python3
"""Counterfactual time-flip pairs from the train window only (t_q <= T_TRAIN) on entities outside the held-out set.
A pair shares entity and template and differs in the anchor; each side has a gold trajectory read from the store view.
Writes out/qa/pairs_<view>.jsonl. `--view append` builds the same pairs on the append-only store (ablation)."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import argparse, collections, datetime as dt, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from agent_env import Env, oracle
ap = argparse.ArgumentParser(); ap.add_argument('--view', default='bitemp'); ap.add_argument('--n', type=int, default=20000); a = ap.parse_args()
OUT = str(_OUT_DIR / 'qa'); rnd = random.Random(7)
T_TRAIN = '2018-07-18'
env = Env(append_only=a.view == 'append')  # Snapshot_State ranks by Hawkes intensity once hawkes_qa.json exists
pairs = json.load(open(f'{OUT}/flip_pairs.json'))
qs = json.load(open(f'{OUT}/questions.json'))
held = set(json.load(open(f'{OUT}/held.json')))
def adays(s, n): return (dt.date.fromisoformat(s) + dt.timedelta(days=n)).isoformat()
def dstr(s):
    d = dt.date.fromisoformat(s); return rnd.choice([d.strftime('%B %-d, %Y'), s, d.strftime('%b %-d, %Y')])
def pick(a0, b0):
    n = (dt.date.fromisoformat(b0) - dt.date.fromisoformat(a0)).days
    return adays(a0, rnd.randrange(n)) if n > 0 else None
QT = {'chief_executive': ['Who was the CEO of {e} on {d}?', 'As of {d}, who served as chief executive of {e}?', 'Name the chief executive officer of {e} at {d}.'],
      'chairperson': ['Who chaired the board of {e} on {d}?', 'As of {d}, who was the chairperson of {e}?'], 'headquarters': ['Where was {e} headquartered on {d}?', 'As of {d}, what was the headquarters location of {e}?'],
      'parent_organization': ['What was the parent organization of {e} on {d}?', 'As of {d}, which group was {e} a subsidiary of?'], 'owner': ['Who owned {e} on {d}?', 'As of {d}, who was the owner of {e}?'],
      'director_manager': ['Who was the director of {e} on {d}?', 'As of {d}, who managed {e}?'], 'registered_name': ['What was the SEC registered name of the registrant with {e} on {d}?', 'As of {d}, under which name was {e} registered with the SEC?']}
MW = {'revenue': 'annual revenue', 'total_assets': 'total assets', 'total_liabilities': 'total liabilities', 'stockholders_equity': "stockholders' equity", 'net_income': 'annual net income'}
def side(p, t):
    if p.get('cumulative'):
        q = f"As of {dstr(t)}, what is the most recently reported {MW[p['rel']]} of {p['ref']} in USD?"; fam = 'metric'
    else:
        q = rnd.choice(QT[p['rel']]).format(e=p['ref'], d=dstr(t)); fam = 'state'
    return dict(family=fam, question=q, ref=p['ref'], rel=p['rel'], t_q=t)
out = open(f'{OUT}/pairs_{a.view}_h.jsonl', 'w'); n = collections.Counter()
ex = [p for p in pairs if not p.get('cumulative')]; cm = [p for p in pairs if p.get('cumulative')]
def emit(fam, q1, m1, q2, m2, y1, y2):
    if m1 and m2 and y1 != y2:
        out.write(json.dumps(dict(family=fam, q1=q1['question'], q2=q2['question'], m1=m1, m2=m2, y_old=y1, y_new=y2), ensure_ascii=False) + '\n'); n[fam] += 1; return True
    return False
def fill(fam, want, gen, tries=20):
    k = 0; t = 0
    while n[fam] < want and t < want * tries:
        t += 1; gen()
# 1) exclusive state flips
def g_state():
    p = rnd.choice(ex); t1, t2 = pick(p['a1'], p['b1']), pick(p['a2'], p['b2'])
    if not t1 or not t2: return
    q1, q2 = side(p, t1), side(p, t2); q1['answer'], q2['answer'] = p['old'], p['new']
    emit('state', q1, oracle(env, q1), q2, oracle(env, q2), p['old'], p['new'])
fill('state', int(a.n * .5), g_state)
# 2) metric flips: a newer filing replaces the latest figure
def g_metric():
    p = rnd.choice(cm); t1, t2 = pick(p['a1'], p['b1']), pick(p['a2'], p['b2'])
    if not t1 or not t2: return
    q1, q2 = side(p, t1), side(p, t2); q1['answer'], q2['answer'] = p['old'], p['new']
    emit('metric', q1, oracle(env, q1), q2, oracle(env, q2), p['old'], p['new'])
fill('metric', int(a.n * .15), g_metric)
# 3) delta / event / lineage / twohop pairs read their answers from the store at two anchors (train window, no held-out entities)
keys_cu = [k for k in env.store.idx if env.store.v[env.store.idx[k][0]]['tau'] == 'Cumulative' and k[0] not in held and len(env.store.idx[k]) >= 4]
keys_tr = [k for k in env.store.idx if k[1] == 'material_event' and k[0] not in held and len(env.store.idx[k]) >= 4]
keys_ex = [k for k in env.store.idx if env.store.v[env.store.idx[k][0]]['tau'] == 'Exclusive' and k[0] not in held and len(env.store.idx[k]) >= 3]
parents = [k for k in env.store.idx if k[1] in ('parent_organization', 'owner') and k[0] not in held]
eidref = lambda e, r: env.name[e]
def g_delta():
    e, r = rnd.choice(keys_cu); ref = env.name[e]
    if env.link(ref) != e: return
    vs = [env.store.v[i] for i in env.store.idx[(e, r)] if env.store.v[i]['tx'] <= T_TRAIN]
    if len(vs) < 4: return
    j = rnd.randrange(2, len(vs) - 1); t1 = adays(vs[j]['tx'], 1); t2 = adays(vs[j + 1]['tx'], 1)
    if t2 > T_TRAIN or t1 >= t2: return
    T = "As of {d}, by how many USD did the reported {m} of {e} change between its two most recent fiscal years? Give a signed number."
    ans = []
    for t in (t1, t2):
        rows = env.store.snapshot_state(e, r, t)
        if not rows or rows[-1].get('delta') is None and not env.append_only: return
        per = {}
        for z in rows: per[z['vfrom']] = z
        ks = sorted(per)
        if len(ks) < 2: return
        ans.append(f"{int(per[ks[-1]]['val'] - per[ks[-2]]['val']):,}")
    q1 = dict(family='delta', question=T.format(d=dstr(t1), m=MW[r], e=ref), ref=ref, rel=r, t_q=t1, answer=ans[0]); q2 = dict(family='delta', question=T.format(d=dstr(t2), m=MW[r], e=ref), ref=ref, rel=r, t_q=t2, answer=ans[1])
    emit('delta', q1, oracle(env, q1), q2, oracle(env, q2), ans[0], ans[1])
fill('delta', int(a.n * .08), g_delta)
def g_event():
    e, r = rnd.choice(keys_tr); ref = env.name[e]
    if env.link(ref) != e: return
    vs = sorted([env.store.v[i] for i in env.store.idx[(e, r)] if env.store.v[i]['tx'] <= T_TRAIN], key=lambda z: (z['tx'], z['id']))
    if len(vs) < 3: return
    j = rnd.randrange(1, len(vs) - 1); t1 = vs[j]['tx']; t2 = vs[j + 1]['tx']
    if t1 >= t2 or vs[j]['value'] == vs[j + 1]['value']: return
    if len({z['value'] for z in vs if z['tx'] == t1}) != 1 or len({z['value'] for z in vs if z['tx'] == t2}) != 1: return
    T = "As of {d}, what kind of material event did {e} disclose most recently in an 8-K?"
    q1 = dict(family='event', question=T.format(d=dstr(t1), e=ref), ref=ref, rel=r, t_q=t1, answer=vs[j]['value']); q2 = dict(family='event', question=T.format(d=dstr(t2), e=ref), ref=ref, rel=r, t_q=t2, answer=vs[j + 1]['value'])
    emit('event', q1, oracle(env, q1), q2, oracle(env, q2), vs[j]['value'], vs[j + 1]['value'])
fill('event', int(a.n * .1), g_event)
RW = {'chief_executive': 'CEOs', 'chairperson': 'chairpersons', 'headquarters': 'headquarters locations', 'parent_organization': 'parent organizations', 'owner': 'owners', 'director_manager': 'directors'}
def g_lin():
    e, r = rnd.choice(keys_ex)
    if r not in RW or env.link(env.name[e]) != e: return
    ref = env.name[e]
    base = env.store.as_of(e, r, T_TRAIN)
    if len(base) < 3: return
    j = rnd.randrange(1, len(base) - 1); t0 = base[j - 1]['vfrom']; t1 = adays(base[j]['vfrom'], 1); t2 = adays(base[j + 1]['vfrom'], 1)
    if t2 > T_TRAIN or t0 >= t1: return
    res = []
    for t in (t1, t2):
        rs = (env.gold if env.append_only else env.store).lineage(e, r, t0, t); v = []
        for z in rs:
            if not v or v[-1] != z['value']: v.append(z['value'])
        res.append(' -> '.join(v))
    if not res[0] or not res[1]: return
    T = "List in chronological order the {w} of {e} between {a} and {b}."
    q1 = dict(family='lineage', question=T.format(w=RW[r], e=ref, a=dstr(t0), b=dstr(t1)), ref=ref, rel=r, t_q=t1, t_start=t0, answer=res[0]); q2 = dict(family='lineage', question=T.format(w=RW[r], e=ref, a=dstr(t0), b=dstr(t2)), ref=ref, rel=r, t_q=t2, t_start=t0, answer=res[1])
    emit('lineage', q1, oracle(env, q1), q2, oracle(env, q2), res[0], res[1])
fill('lineage', int(a.n * .09), g_lin)
def g_two():
    e, r = rnd.choice(parents); ref = env.name[e]
    if env.link(ref) != e: return
    ps = [z for z in env.store.as_of(e, r, T_TRAIN)]
    if not ps: return
    pz = rnd.choice(ps); pe = env.link(pz['value'])
    if not pe or pe in held: return
    r2 = rnd.choice(['chief_executive', 'chairperson'])
    v2 = env.store.as_of(pe, r2, T_TRAIN)
    if len(v2) < 2: return
    j = rnd.randrange(len(v2) - 1); end = pz['vto'] or T_TRAIN
    if not (pz['vfrom'] < v2[j + 1]['vfrom'] < end): return
    t1 = adays(v2[j + 1]['vfrom'], -1); t2 = adays(v2[j + 1]['vfrom'], rnd.randrange(1, 120))
    if t1 < max(v2[j]['vfrom'], pz['vfrom']) or t2 > T_TRAIN or t2 >= end: return
    T = "As of {d}, who was the {w} of the {o} of {e}?"
    w = 'CEO' if r2 == 'chief_executive' else 'chairperson'; o = 'parent organization' if r == 'parent_organization' else 'owner'
    q1 = dict(family='twohop', question=T.format(d=dstr(t1), w=w, o=o, e=ref), ref=ref, rel=r, rel2=r2, t_q=t1, answer=v2[j]['value']); q2 = dict(family='twohop', question=T.format(d=dstr(t2), w=w, o=o, e=ref), ref=ref, rel=r, rel2=r2, t_q=t2, answer=v2[j + 1]['value'])
    emit('twohop', q1, oracle(env, q1), q2, oracle(env, q2), v2[j]['value'], v2[j + 1]['value'])
fill('twohop', int(a.n * .08), g_two, tries=200)
print(n, sum(n.values()), flush=True)
out.close()

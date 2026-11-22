#!/usr/bin/env python3
"""Tool environment and prompts of the FinChrono agent. Every read is bound to the t_q the agent passes; the store enforces the snapshot."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import bisect, json, math, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from store import BiTemporalStore
ROOT = str(_OUT_DIR / 'qa')
RELS = ['chief_executive', 'chairperson', 'headquarters', 'parent_organization', 'owner', 'director_manager', 'registered_name', 'revenue', 'total_assets', 'total_liabilities', 'stockholders_equity', 'net_income', 'material_event']
SYSTEM = """You are a financial temporal memory agent. Answer from the memory tools only, never from prior knowledge. Reply with one JSON object per turn.
Tools:
{"tool": "Snapshot_State", "entity": <name or "CIK ..." as in the question>, "rel": <relation>, "t_q": "YYYY-MM-DD"}
{"tool": "Trace_State_Lineage", "entity": ..., "rel": ..., "t_start": "YYYY-MM-DD", "t_q": "YYYY-MM-DD"}
{"tool": "Propagate_Hawkes_Shock", "source_entity": ..., "t_q": "YYYY-MM-DD", "max_hops": <1-3>}
{"tool": "Emit_Temporal_Verdict", "answer": <answer>, "cited_event_ids": [<ids>], "validity_interval": [<start>, <end or null>]} ends the episode.
Relations: """ + ', '.join(RELS) + """.
Tags: ACTIVE holds at t_q, SUPERSEDED no longer holds, EVENT is a point event. t_q is the date in the question.
Write lists as "A -> B -> C" in time order; amounts exactly as recorded; changes with a sign."""
ZS_HINT = '\nThink about which tool gives the needed record. Output only the JSON object.'

class HawkesRank:
    """lambda(e_i | t_q) = exp(-beta_r (t_q - t_i)) + sum_{e_j earlier, same entity} kappa[r_j, r_i] exp(-beta[r_j] (t_i - t_j)) (scaled); ablations: nomutual, nodecay"""
    def __init__(self, df, abl=()):
        h = json.load(open(f'{ROOT}/hawkes_qa.json')); self.t = {x: i for i, x in enumerate(h['types'])}
        self.b = h['beta']; self.k = h['kappa']; self.abl = set(abl); self.hist = {}
        for r in df[(df.tau != 'Exclusive') | df.extra.fillna('').str.contains('"open"')][['eid', 'rel', 'tau', 'value', 'tx']].itertuples(index=False):
            self.hist.setdefault(r.eid, []).append((r.tx, self.ty(r.tau, r.value, r.rel)))
        for k in self.hist:
            self.hist[k].sort(key=lambda x: x[0])
        self.mx = max(max(x) for x in self.k) or 1.0; self.cache = {}
    def ty(self, tau, value, rel):
        return self.t.get(value if tau == 'Transient' else rel)
    def __call__(self, z, tq, eid):
        import datetime as dt
        d = lambda s: dt.date.fromisoformat(s).toordinal()
        ri = self.ty(z['tau'], z['value'], z['rel']); dq = max(0, d(tq) - d(z['tx']))
        beta = 0.05 if 'nodecay' in self.abl else (self.b[ri] if ri is not None else 0.01)
        lam = math.exp(-beta * dq)
        if 'nomutual' not in self.abl and ri is not None:
            key = (eid, z['id'])
            if key not in self.cache:
                ti = d(z['tx']); s = 0.0
                h = self.hist.get(eid, ())
                for tx, rj in h[max(0, bisect.bisect_left(h, (z['tx'],)) - 400):bisect.bisect_left(h, (z['tx'],))]:   # strictly before the candidate
                    if rj is None or ti - d(tx) > 365:
                        continue
                    s += self.k[rj][ri] / self.mx * math.exp(-self.b[rj] * (ti - d(tx)))
                self.cache[key] = s
            lam = lam + self.cache[key] * math.exp(-beta * dq)
        return lam

class Env:
    def __init__(self, append_only=False, K=6, abl=(), rank='hawkes', drop=0.0, delay=0.0, delay_days=30, seed=0):
        self.df = pd.read_parquet(f'{ROOT}/stream.parquet')
        if drop or delay:          # degraded stream: lost events and late filings; gold answers stay those of the clean timeline
            import numpy as np
            rs = np.random.RandomState(seed); df = self.df
            df = df[rs.rand(len(df)) >= drop].copy()
            late = rs.rand(len(df)) < delay; extra = pd.to_timedelta(rs.randint(1, delay_days + 1, len(df)), unit='D')
            ok = df.tx >= '1700-01-01'; tx = pd.to_datetime(df.tx.where(ok), errors='coerce') + extra
            df.loc[late & ok, 'tx'] = tx[late & ok].dt.strftime('%Y-%m-%d')
            self.df = df.sort_values(['tx', 'id']).reset_index(drop=True)
        intensity = None
        if rank == 'hawkes' and os.path.exists(f'{ROOT}/hawkes_qa.json'):
            intensity = HawkesRank(self.df, abl)
        self.store = BiTemporalStore(append_only=append_only); self.store.ingest(self.df)
        self.append_only = append_only; self.K = K; self.intensity = intensity
        if append_only:            # supervision for the append-only variant still comes from the true timeline
            self.gold = BiTemporalStore(); self.gold.ingest(self.df)
        e = pd.read_parquet(f'{ROOT}/entities.parquet')
        self.by_name = {}
        for r in e.itertuples():
            self.by_name[r.name.lower()] = r.eid
            if r.cik and isinstance(r.cik, str):
                self.by_name['cik ' + r.cik.zfill(10)] = r.eid
        self.name = dict(zip(e.eid, e.name))
    def link(self, s):
        k = str(s).strip().lower()
        if k in self.by_name:
            return self.by_name[k]
        m = re.search(r'cik:?\s*(\d{4,10})', k)       # an entity written with its CIK in brackets
        k2 = re.sub(r'\s*\([^)]*\)\s*$', '', k).strip()
        if k2 in self.by_name:
            return self.by_name[k2]
        return self.by_name.get('cik ' + m.group(1).zfill(10)) if m else None
    def fmt(self, z, ename):
        base = f"[{z['id']}] {z['value']}" if self.append_only else f"[{z['id']}|{z['state']}] {z['value']}"
        if z['tau'] == 'Exclusive':
            return base + (f"; since {z['vfrom']}" if self.append_only else f"; {z['vfrom']}..{z['vto'] or 'open'}")
        if z['tau'] == 'Cumulative':
            txt = base + f"; FY end {z['vfrom']}; filed {z['tx']}"
            if z.get('delta') is not None and not self.append_only:
                txt += f"; change {int(z['delta']):+,}"
            return txt
        return base + f"; dated {z['vfrom']}; filed {z['tx']}"
    def _rank(self, rows, tq, eid=None):
        """keep ACTIVE rows plus the top intensity rows; display chronologically (state machine) or by intensity (append-only)."""
        if not rows:
            return rows
        if self.intensity:
            sc = {z['id']: (self.intensity(z, tq, eid), z['tx'], z['id']) for z in rows}
        else:
            sc = {z['id']: (z['tx'], z['id']) for z in rows}
        order = sorted(rows, key=lambda z: sc[z['id']], reverse=True)
        keep = [z for z in rows if z['state'] == 'ACTIVE']
        for z in order:
            if len(keep) >= self.K:
                break
            if z not in keep:
                keep.append(z)
        if self.append_only:
            return sorted(keep, key=lambda z: sc[z['id']], reverse=True)
        return sorted(keep, key=lambda z: (z['vfrom'], z['tx'], z['id']))
    def snapshot(self, entity, rel, tq):
        eid = self.link(entity)
        if eid is None:
            return f'No entity named {entity}.', []
        if rel not in RELS:
            return f'Unknown relation {rel}.', []
        rows = self._rank(self.store.snapshot_state(eid, rel, tq), tq, eid)
        if not rows:
            return f'No {rel} records for {entity} visible at {tq}.', []
        return '\n'.join(self.fmt(z, entity) for z in rows), rows
    def lineage(self, entity, rel, t0, tq):
        eid = self.link(entity)
        if eid is None:
            return f'No entity named {entity}.', []
        rows = self.store.lineage(eid, rel, t0, tq)
        if self.append_only:
            rows = [z for z in self.store.as_of(eid, rel, tq)]
        rows = rows[-self.K * 2:]
        if not rows:
            return f'No {rel} records for {entity} between {t0} and {tq}.', []
        return '\n'.join(self.fmt(z, entity) for z in rows), rows
    def shock(self, entity, tq, hops=2):
        """Propagate_Hawkes_Shock over corporate links visible at tq: linked companies ranked by their decayed adverse 8-K events."""
        eid = self.link(entity)
        if eid is None:
            return f'No entity named {entity}.'
        seen = {eid: 0}; fr = [eid]
        for h in range(1, int(hops) + 1):
            nf = []
            for e in fr:
                for rel in ('parent_organization', 'owner'):
                    for z in self.store.snapshot_state(e, rel, tq):
                        if z['state'] == 'ACTIVE' and self.link(z['value']) and self.link(z['value']) not in seen:
                            seen[self.link(z['value'])] = h; nf.append(self.link(z['value']))
            fr = nf
        seen.pop(eid); bad = {'bankruptcy or receivership', 'debt acceleration trigger', 'delisting notice', 'non-reliance on prior financial statements', 'material impairment', 'cybersecurity incident'}
        rows = []
        for e, h in seen.items():
            ev = [z for z in self.store.as_of(e, 'material_event', tq) if z['value'] in bad]
            rows.append((len(ev), self.name.get(e, e), h))
        rows.sort(reverse=True)
        return '\n'.join(f'{n} (hop {h}): {c} adverse events' for c, n, h in rows[:self.K]) or f'No linked companies of {entity} visible at {tq}.'
    def step(self, a, cap=None):
        """a: parsed tool call; cap: the anchor of the question. The store never serves a read later than the anchor, whatever date the model passes."""
        tool = a.get('tool')
        def when(k):
            t = str(a[k])[:10]
            return min(t, cap) if cap else t
        try:
            if tool == 'Snapshot_State':
                return self.snapshot(a['entity'], a['rel'], when('t_q'))[0]
            if tool == 'Trace_State_Lineage':
                return self.lineage(a['entity'], a['rel'], str(a['t_start'])[:10], when('t_q'))[0]
            if tool == 'Propagate_Hawkes_Shock':
                return self.shock(a.get('source_entity'), when('t_q'), a.get('max_hops', 2))
        except KeyError as e:
            return f'Missing argument {e}.'
        return 'Unknown tool.'

def verdict(answer, ids, interval):
    return json.dumps({'tool': 'Emit_Temporal_Verdict', 'answer': answer, 'cited_event_ids': ids, 'validity_interval': interval}, ensure_ascii=False)
def call(tool, **kw):
    return json.dumps({'tool': tool, **kw}, ensure_ascii=False)

def parse_json(txt):
    """lenient: a verdict with an unquoted amount such as "answer": 204,358,000 is still read as an answer"""
    txt = txt.strip()
    m = re.search(r'\{.*\}', txt, re.S)
    if not m:
        return None
    body = m.group(0)
    try:
        return json.loads(body)
    except Exception:
        if 'Emit_Temporal_Verdict' in body:
            am = re.search(r'"answer"\s*:\s*("(?:[^"\\]|\\.)*"|\[.*?\]|[^,{}\[\]"]+(?:,\d{3})*)\s*,\s*"(?:cited|validity)', body, re.S)
            ids = re.search(r'"cited_event_ids"\s*:\s*\[(.*?)\]', body, re.S)
            if am:
                ans = am.group(1).strip()
                try:
                    ans = json.loads(ans)
                except Exception:
                    pass
                return {'tool': 'Emit_Temporal_Verdict', 'answer': ans, 'cited_event_ids': re.findall(r'\d+', ids.group(1)) if ids else []}
        return None

def answer_text(a):
    """lists become 'A -> B'; everything else a string"""
    if isinstance(a, (list, tuple)):
        return ' -> '.join(str(x) for x in a)
    return '' if a is None else str(a)

def oracle(env, q):
    """gold trajectory built by reading the store (same view the agent sees). Returns messages or None if the store view cannot support the answer."""
    fam, tq = q['family'], q['t_q']; ref = q.get('ref') or q.get('entity_ref')
    M = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'Question: ' + q['question']}]
    def act(tool, **kw):
        c = call(tool, **kw); M.append({'role': 'assistant', 'content': c}); return c
    if fam in ('state', 'metric', 'delta', 'event'):
        act('Snapshot_State', entity=ref, rel=q['rel'], t_q=tq)
        obs, rows = env.snapshot(ref, q['rel'], tq); M.append({'role': 'user', 'content': 'Observation:\n' + obs})
        if fam == 'event':
            rows = sorted(rows, key=lambda z: (z['tx'], z['id'])); z = rows[-1] if rows else None
            ans = z['value'] if z else None; ids = [z['id']] if z else []; iv = [z['vfrom'], z['vfrom']] if z else None
        else:
            act_rows = [z for z in rows if z['state'] == 'ACTIVE'] if not env.append_only else sorted(rows, key=lambda z: (z['vfrom'], z['tx'], z['id']))[-1:]
            if len(act_rows) != 1:
                return None
            z = act_rows[0]; ids = [z['id']]; iv = [z['vfrom'], z['vto']]
            if fam == 'delta':
                if env.append_only:
                    per = sorted(rows, key=lambda x: (x['vfrom'], x['tx'], x['id'])); bypd = {}
                    for x in per: bypd[x['vfrom']] = x
                    ks = sorted(bypd)
                    if len(ks) < 2: return None
                    ans = f"{int(bypd[ks[-1]]['val'] - bypd[ks[-2]]['val']):,}"; ids = [bypd[ks[-1]]['id'], bypd[ks[-2]]['id']]
                else:
                    if z.get('delta') is None: return None
                    ans = f"{int(z['delta']):,}"
                    prev = [x for x in rows if x['vfrom'] < z['vfrom']]
                    ids = [z['id']] + ([prev[-1]['id']] if prev else [])
            else:
                ans = z['value']
    elif fam == 'lineage':
        act('Trace_State_Lineage', entity=ref, rel=q['rel'], t_start=q['t_start'], t_q=tq)
        obs, rows = env.lineage(ref, q['rel'], q['t_start'], tq); M.append({'role': 'user', 'content': 'Observation:\n' + obs})
        if env.append_only:        # the agent sees every record; the target is the true lineage inside [t_start, t_q]
            rows = env.gold.lineage(env.link(ref), q['rel'], q['t_start'], tq)
        vals = []
        for z in rows:
            if not vals or vals[-1] != z['value']: vals.append(z['value'])
        ans = ' -> '.join(vals); ids = [z['id'] for z in rows]; iv = [q['t_start'], tq]
    elif fam == 'twohop':
        act('Snapshot_State', entity=ref, rel=q['rel'], t_q=tq)
        obs, rows = env.snapshot(ref, q['rel'], tq); M.append({'role': 'user', 'content': 'Observation:\n' + obs})
        a = [z for z in rows if z['state'] == 'ACTIVE'] if not env.append_only else sorted(rows, key=lambda z: (z['vfrom'], z['tx'], z['id']))[-1:]
        if len(a) != 1:
            return None
        act('Snapshot_State', entity=a[0]['value'], rel=q['rel2'], t_q=tq)
        obs, rows2 = env.snapshot(a[0]['value'], q['rel2'], tq); M.append({'role': 'user', 'content': 'Observation:\n' + obs})
        b = [z for z in rows2 if z['state'] == 'ACTIVE'] if not env.append_only else sorted(rows2, key=lambda z: (z['vfrom'], z['tx'], z['id']))[-1:]
        if len(b) != 1:
            return None
        ans = b[0]['value']; ids = [a[0]['id'], b[0]['id']]; iv = [max(a[0]['vfrom'], b[0]['vfrom']), None]
    else:
        return None
    if ans is None or ans != q['answer']:
        return None
    M.append({'role': 'assistant', 'content': verdict(ans, ids, iv)})
    return M

#!/usr/bin/env python3
"""Verify store snapshots against independently built gold answers, and measure ingest / read throughput."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import json, time, sys, collections
import pandas as pd
sys.path.insert(0, _os.path.join(_SRC_DIR, 'qa'))
from store import BiTemporalStore
OUT = str(_OUT_DIR / 'qa')
df = pd.read_parquet(f'{OUT}/stream.parquet'); qs = json.load(open(f'{OUT}/questions.json'))
s = BiTemporalStore(); t = s.ingest(df); print('ingest ev/s', len(df) / t, 'versions', len(s.v), flush=True)
ok = collections.Counter(); tot = collections.Counter(); bad = []
t0 = time.time(); n = 0
for q in qs:
    tot[q['family']] += 1; n += 1
    if q['family'] in ('state', 'metric'):
        rows = s.snapshot_state(q['eid'], q['rel'], q['t_q']); act = [z for z in rows if z['state'] == 'ACTIVE']
        hit = len(act) == 1 and act[0]['value'] == q['answer']
    elif q['family'] == 'event':
        rows = s.as_of(q['eid'], q['rel'], q['t_q']); rows.sort(key=lambda z: (z['tx'], z['id'])); hit = bool(rows) and rows[-1]['value'] == q['answer']
    elif q['family'] == 'lineage':
        rows = s.lineage(q['eid'], q['rel'], q['t_start'], q['t_q']); v = []
        for z in rows:
            if not v or v[-1] != z['value']: v.append(z['value'])
        hit = v == q['answer_list']
    elif q['family'] == 'delta':
        rows = s.snapshot_state(q['eid'], q['rel'], q['t_q']); hit = bool(rows) and rows[-1]['delta'] is not None and f"{int(rows[-1]['delta']):,}" == q['answer']
    elif q['family'] == 'twohop':
        act = [z for z in s.snapshot_state(q['eid2'], q['rel2'], q['t_q']) if z['state'] == 'ACTIVE']; hit = len(act) == 1 and act[0]['value'] == q['answer']
    ok[q['family']] += hit
    if not hit and len(bad) < 12:
        bad.append((q['family'], q['question'], q['answer'], [(z['value'], z['vfrom'], z['vto'], z['state']) for z in rows][-5:]))
print('read ms/q', 1000 * (time.time() - t0) / n)
print({k: round(ok[k] / tot[k], 4) for k in tot}, tot)
for b in bad: print(b)

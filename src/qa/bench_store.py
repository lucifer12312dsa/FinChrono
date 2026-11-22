#!/usr/bin/env python3
"""Store costs: ingest rate, as-of latency of the in-memory index vs the same bi-temporal query in DuckDB, state-machine overhead vs append-only."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
_TMP_DIR.mkdir(parents=True, exist_ok=True)
import json, os, random, sys, time
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import BiTemporalStore
O = str(_OUT_DIR / 'qa'); R = {}
df = pd.read_parquet(f'{O}/stream.parquet'); qs = json.load(open(f'{O}/questions.json'))
for name, ap in (('bitemporal', False), ('append', True)):
    s = BiTemporalStore(append_only=ap); t = time.time(); s.ingest(df); R[f'{name}_ingest_ev_s'] = len(df) / (time.time() - t); R[f'{name}_versions'] = len(s.v)
    if not ap:
        S = s
s = S; p = str(_TMP_DIR / 'bench.duckdb')
if os.path.exists(p):
    os.remove(p)
t = time.time(); s.to_duckdb(p); R['duckdb_load_s'] = time.time() - t; con = duckdb.connect(p)
Q = [(q['eid'], q['rel'], q['t_q']) for q in qs if q['family'] in ('state', 'metric', 'event')]; random.Random(0).shuffle(Q); Q = Q[:2000]
t = time.time()
for q in Q:
    s.as_of(*q)
R['memory_asof_ms'] = 1000 * (time.time() - t) / len(Q)
sql = "SELECT id, value, vfrom, CASE WHEN tau = 'Transient' THEN 'EVENT' WHEN closed_tx IS NOT NULL AND closed_tx <= ? AND vto <= ? THEN 'SUPERSEDED' ELSE 'ACTIVE' END AS st FROM versions WHERE eid = ? AND rel = ? AND tx <= ? AND vfrom <= ? ORDER BY vfrom, tx, id"
t = time.time()
for e, r, tq in Q:
    con.execute(sql, [tq, tq, e, r, tq, tq]).fetchall()
R['duckdb_asof_ms'] = 1000 * (time.time() - t) / len(Q)
R['n_events'] = len(df); R['rows_per_read_mean'] = float(np.mean([len(s.as_of(*q)) for q in Q]))
json.dump(R, open(str(_OUT_DIR / 'eval' / 'store_bench.json'), 'w'), indent=1); print(json.dumps(R), flush=True)

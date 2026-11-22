#!/usr/bin/env python3
"""Bi-temporal MVCC state-transition store (DuckDB base table + in-memory adjacency index).
Each event of the stream is one transaction ordered by transaction time. Version rows keep:
  vfrom/vto   : valid-time interval, closed by the state machine or by an explicit end event
  tx / closed_tx : transaction time at which the version was written / closed
  prev / next : SUPERSEDED_BY version chain (Exclusive) or evolution list with delta (Cumulative)
`as_of(entity, rel, tq)` returns the snapshot visible at tq: only versions with tx <= tq and vfrom <= tq;
a version closed after tq is reported ACTIVE (its closure is not yet visible).
With `append_only=True` the state machine is off: nothing is closed and no chain is kept (ablation)."""
import bisect, collections, json, time
import pandas as pd

class BiTemporalStore:
    def __init__(self, append_only=False):
        self.append_only = append_only
        self.v = {}                                   # version id -> record
        self.idx = collections.defaultdict(list)      # (eid, rel) -> version ids in write order
        self.active = {}                              # (eid, rel) -> {value: vid} for Exclusive
        self.open_stmts = {}                          # (eid, rel) -> {value: open source statements}
        self.pending_close = {}                       # ((eid, rel), value) -> ends that arrived before their opening
        self.txn = 0

    def ingest(self, df):
        """df sorted by tx; columns of stream.parquet. Returns seconds spent."""
        t0 = time.time()
        for r in df.itertuples(index=False):
            self.write(r)
        return time.time() - t0

    def write(self, r):
        self.txn += 1
        ex = json.loads(r.extra) if isinstance(r.extra, str) else {}
        k = (r.eid, r.rel)
        if r.tau == 'Exclusive':
            if self.append_only:
                if ex.get('kind') == 'close':
                    return
                self._new(r, k)
                return
            cur = self.active.setdefault(k, {})
            stmts = self.open_stmts.setdefault(k, {})
            if ex.get('kind') == 'close':          # explicit end of one source statement; the state ends with its last open statement
                held = stmts.get(r.value)
                if held is not None:
                    held.discard(ex.get('stmt'))
                    if held:
                        return
                vid = cur.pop(r.value, None)
                if vid is not None:
                    self._close(vid, r.vfrom, r.tx, None)
                    return
                # not active: an end that arrives after the state was superseded, or before its opening
                same = [self.v[i] for i in self.idx[k] if self.v[i]['value'] == r.value and self.v[i]['vfrom'] <= r.vfrom]
                if same:
                    z = max(same, key=lambda x: (x['vfrom'], x['id']))
                    if z['vto'] is None or r.vfrom < z['vto']:
                        self._close(z['id'], r.vfrom, r.tx, None)
                else:
                    self.pending_close.setdefault((k, r.value), []).append((r.vfrom, r.tx))
                return
            if any(self.v[i]['vfrom'] > r.vfrom for i in self.idx[k]):
                self._late_exclusive(r, k)         # out-of-order arrival: a newer state is already known, active or not
                return
            for val, vid in list(cur.items()):     # transition: close whatever else is active
                if val != r.value:
                    self._close(vid, r.vfrom, r.tx, r.id)
                    del cur[val]; stmts.pop(val, None)
            stmts.setdefault(r.value, set()).add(ex.get('stmt'))
            if r.value in cur:                     # re-assertion of an active state keeps the open version
                return
            vid = self._new(r, k)
            cur[r.value] = vid
            self._apply_pending(k, r.value, vid, r.tx, cur)
        elif r.tau == 'Cumulative':
            ix = self.idx[k]
            vid = self._new(r, k)
            if not self.append_only:
                prior = [i for i in ix[:-1] if self.v[i]['vfrom'] < r.vfrom]
                same = [i for i in ix[:-1] if self.v[i]['vfrom'] == r.vfrom]
                if same:                           # restatement of the same fiscal period closes the old version
                    self._close(same[-1], r.tx, r.tx, r.id)
                if prior:
                    best = max(prior, key=lambda i: (self.v[i]['vfrom'], self.v[i]['tx'], i))
                    self.v[vid]['prev'] = best
                    self.v[vid]['delta'] = ex['val'] - self.v[best]['val']
                self.v[vid]['val'] = ex['val']
            else:
                self.v[vid]['val'] = ex['val']
        else:
            self._new(r, k)

    def _late_exclusive(self, r, k):
        """Insert a state that arrives after a newer one: what is active stays, the late version is closed at its successor's valid-from, and the version
        it follows gets a new closure at this transaction. Older closures remain, so a read before r.tx still sees the history as it was then known."""
        vs = sorted((self.v[i] for i in self.idx[k]), key=lambda z: (z['vfrom'], z['id']))
        succ = next(z for z in vs if z['vfrom'] > r.vfrom)
        prev = [z for z in vs if z['vfrom'] <= r.vfrom]
        vid = self._new(r, k)
        self._close(vid, succ['vfrom'], r.tx, succ['id'])
        # the predecessor's end moves back only if a successor had closed it; an explicit earlier end stands
        if prev and prev[-1]['value'] != r.value and (prev[-1]['vto'] is None or (prev[-1]['vto'] > r.vfrom and prev[-1]['succ'] is not None)):
            self._close(prev[-1]['id'], r.vfrom, r.tx, vid)
        self._apply_pending(k, r.value, vid, r.tx, None)

    def _apply_pending(self, k, value, vid, tx, cur):
        ends = [e for e in self.pending_close.get((k, value), []) if e[0] >= self.v[vid]['vfrom']]
        if ends:
            self._close(vid, min(e[0] for e in ends), tx, None)
            self.pending_close[(k, value)] = [e for e in self.pending_close[(k, value)] if e not in ends]
            if cur is not None:
                cur.pop(value, None)

    def _new(self, r, k):
        vid = r.id
        self.v[vid] = dict(id=vid, eid=r.eid, rel=r.rel, tau=r.tau, value=r.value, vfrom=r.vfrom, vto=None, tx=r.tx, closed_tx=None, succ=None, prev=None, delta=None, text=r.text, closures=[])
        self.idx[k].append(vid)
        return vid

    def _close(self, vid, vto, tx, succ):
        x = self.v[vid]; x['vto'] = vto; x['closed_tx'] = tx; x['succ'] = succ; x['closures'].append((tx, vto))

    # ---------- snapshot reads ----------
    def as_of(self, eid, rel, tq):
        """All versions of (eid,rel) visible at tq with their state as seen at tq. Sorted by valid-from."""
        out = []
        for vid in self.idx.get((eid, rel), ()):
            x = self.v[vid]
            if x['tx'] > tq or x['vfrom'] > tq:
                continue
            seen = [c for c in x['closures'] if c[0] <= tq]    # the latest closure already written at tq
            vto = seen[-1][1] if seen else None
            closed = vto is not None and vto <= tq
            if self.append_only:
                st = 'UNTAGGED'
            elif x['tau'] == 'Transient':
                st = 'EVENT'
            else:
                st = 'SUPERSEDED' if closed else 'ACTIVE'
            out.append(dict(id=vid, value=x['value'], vfrom=x['vfrom'], vto=(vto if closed else None), tx=x['tx'], state=st, delta=x['delta'] if not self.append_only else None, val=x.get('val'), text=x['text'], tau=x['tau'], rel=x['rel']))
        out.sort(key=lambda z: (z['vfrom'], z['tx'], z['id']))
        return out

    def snapshot_state(self, eid, rel, tq):
        rows = self.as_of(eid, rel, tq)
        if rows and rows[0]['tau'] == 'Cumulative' and not self.append_only:
            best = {}
            for z in rows:
                best[z['vfrom']] = z                # last visible report per fiscal period
            rows = [best[k] for k in sorted(best)]
            for z in rows[:-1]:
                z['state'] = 'SUPERSEDED'
            if rows:
                rows[-1]['state'] = 'ACTIVE'
            # delta must be taken against the previous period version visible at tq
            for a, b in zip(rows, rows[1:]):
                b['delta'] = b['val'] - a['val'] if b['val'] is not None and a['val'] is not None else None
            rows[0]['delta'] = None
        return rows

    def lineage(self, eid, rel, t_start, tq):
        rows = self.snapshot_state(eid, rel, tq)
        return [z for z in rows if (z['vto'] is None or z['vto'] > t_start)]

    def to_duckdb(self, path):
        import duckdb
        con = duckdb.connect(path)
        df = pd.DataFrame([{k: x.get(k) for k in ('id', 'eid', 'rel', 'tau', 'value', 'vfrom', 'vto', 'tx', 'closed_tx', 'succ', 'prev', 'delta')} for x in self.v.values()])
        con.register('d', df); con.execute('CREATE OR REPLACE TABLE versions AS SELECT * FROM d')
        con.execute('CREATE INDEX IF NOT EXISTS ix ON versions(eid, rel)'); con.close()

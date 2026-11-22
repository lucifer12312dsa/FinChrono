#!/usr/bin/env python3
"""Unit tests of the bi-temporal state machine on small synthetic histories (no data needed)."""
import json, os, sys
from types import SimpleNamespace as N
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'qa'))
from store import BiTemporalStore
i = [0]
def ev(rel, tau, v, vfrom, tx=None, kind='open', val=None):
    i[0] += 1
    ex = {'kind': kind} if tau == 'Exclusive' else ({'val': val} if tau == 'Cumulative' else {})
    return N(id=i[0], eid='E', rel=rel, tau=tau, value=v, vfrom=vfrom, tx=tx or vfrom, text='', extra=json.dumps(ex))
def act(s, rel, t):
    return [z['value'] for z in s.snapshot_state('E', rel, t) if z['state'] == 'ACTIVE']
# transition closes the old edge; the snapshot before the flip does not see the closure
s = BiTemporalStore(); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2000-01-01'), ev('ceo', 'Exclusive', 'B', '2010-01-01')]]
assert act(s, 'ceo', '2005-01-01') == ['A'], 'old state before flip'
assert act(s, 'ceo', '2015-01-01') == ['B'], 'new state after flip'
assert [z['state'] for z in s.snapshot_state('E', 'ceo', '2015-01-01')] == ['SUPERSEDED', 'ACTIVE']
assert s.snapshot_state('E', 'ceo', '2005-01-01')[0]['vto'] is None, 'closure in the future is invisible'
assert s.snapshot_state('E', 'ceo', '1999-01-01') == [], 'nothing before first state'
# explicit end
s = BiTemporalStore(); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2000-01-01'), ev('ceo', 'Exclusive', 'A', '2005-01-01', kind='close')]]
assert act(s, 'ceo', '2006-01-01') == [] and act(s, 'ceo', '2004-01-01') == ['A']
# out-of-order arrival: B (valid 2005) is filed in 2012 after C (valid 2010)
s = BiTemporalStore(); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2000-01-01'), ev('ceo', 'Exclusive', 'C', '2010-01-01'), ev('ceo', 'Exclusive', 'B', '2005-01-01', tx='2012-01-01')]]
assert act(s, 'ceo', '2013-01-01') == ['C'], 'late state must not replace the newer active one'
assert [z['value'] for z in s.snapshot_state('E', 'ceo', '2013-01-01')] == ['A', 'B', 'C']
assert [z['value'] for z in s.snapshot_state('E', 'ceo', '2011-01-01')] == ['A', 'C'], 'transaction time hides the late filing'
h = {z['value']: z['vto'] for z in BiTemporalStore.as_of(s, 'E', 'ceo', '2011-01-01')}
assert h['A'] == '2010-01-01', 'a later write must not change what a read at 2011 saw'
h = {z['value']: z['vto'] for z in s.as_of('E', 'ceo', '2013-01-01')}
assert h['A'] == '2005-01-01' and h['B'] == '2010-01-01'
# late state whose successor has already ended must not become active
s = BiTemporalStore(); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2000-01-01'), ev('ceo', 'Exclusive', 'C', '2010-01-01'), ev('ceo', 'Exclusive', 'C', '2011-01-01', kind='close'), ev('ceo', 'Exclusive', 'B', '2005-01-01', tx='2012-01-01')]]
assert act(s, 'ceo', '2013-01-01') == [] and [z['vto'] for z in s.as_of('E', 'ceo', '2013-01-01') if z['value'] == 'B'] == ['2010-01-01']
# two statements assert A; A ends only with the last one
def st(v, d, kind, stmt):
    e = ev('ceo', 'Exclusive', v, d, kind=kind); e.extra = json.dumps({'kind': kind, 'stmt': stmt}); return e
s = BiTemporalStore(); [s.write(e) for e in [st('A', '2000-01-01', 'open', 's1'), st('A', '2005-01-01', 'open', 's2'), st('A', '2010-01-01', 'close', 's1'), st('A', '2020-01-01', 'close', 's2')]]
assert act(s, 'ceo', '2015-01-01') == ['A'] and act(s, 'ceo', '2021-01-01') == []
# an end that arrives after the state was superseded moves its end back; an end before its opening is kept for it
s = BiTemporalStore(); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2000-01-01'), ev('ceo', 'Exclusive', 'B', '2010-01-01'), ev('ceo', 'Exclusive', 'A', '2005-01-01', tx='2011-01-01', kind='close')]]
assert [z['vto'] for z in s.as_of('E', 'ceo', '2012-01-01') if z['value'] == 'A'] == ['2005-01-01'] and [z['vto'] for z in s.as_of('E', 'ceo', '2010-06-01') if z['value'] == 'A'] == ['2010-01-01']
s = BiTemporalStore(); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2005-01-01', kind='close'), ev('ceo', 'Exclusive', 'A', '2000-01-01', tx='2006-01-01')]]
assert act(s, 'ceo', '2007-01-01') == []
# a late state does not extend a predecessor with an earlier explicit end
s = BiTemporalStore(); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2000-01-01'), ev('ceo', 'Exclusive', 'A', '2002-01-01', kind='close'), ev('ceo', 'Exclusive', 'C', '2010-01-01'), ev('ceo', 'Exclusive', 'B', '2005-01-01', tx='2012-01-01')]]
assert [z['vto'] for z in s.as_of('E', 'ceo', '2013-01-01') if z['value'] == 'A'] == ['2002-01-01']
# cumulative: valid time vs filing time, delta, restatement
s = BiTemporalStore(); [s.write(e) for e in [ev('rev', 'Cumulative', '100', '2018-12-31', tx='2019-02-01', val=100), ev('rev', 'Cumulative', '130', '2019-12-31', tx='2020-02-01', val=130), ev('rev', 'Cumulative', '120', '2019-12-31', tx='2020-06-01', val=120)]]
assert act(s, 'rev', '2020-01-15') == ['100'], 'period ended but not yet filed'
assert act(s, 'rev', '2020-03-01') == ['130']; assert act(s, 'rev', '2020-07-01') == ['120'], 'restatement'
assert s.snapshot_state('E', 'rev', '2020-03-01')[-1]['delta'] == 30 and s.snapshot_state('E', 'rev', '2020-07-01')[-1]['delta'] == 20
# transient: filing date controls visibility
s = BiTemporalStore(); s.write(ev('event', 'Transient', 'x', '2020-01-01', tx='2020-01-05'))
assert s.as_of('E', 'event', '2020-01-03') == [] and len(s.as_of('E', 'event', '2020-01-06')) == 1
# append-only ablation keeps everything untagged
s = BiTemporalStore(append_only=True); [s.write(e) for e in [ev('ceo', 'Exclusive', 'A', '2000-01-01'), ev('ceo', 'Exclusive', 'B', '2010-01-01')]]
assert {z['state'] for z in s.as_of('E', 'ceo', '2015-01-01')} == {'UNTAGGED'}
print('store tests ok')

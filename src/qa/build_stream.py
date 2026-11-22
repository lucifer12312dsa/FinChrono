#!/usr/bin/env python3
"""Fin-ChronoQA event stream from real sources.
 Exclusive  : Wikidata CEO/chair/HQ/parent/owner/director statements (with start/end qualifiers), SEC registered-name history
 Cumulative : SEC XBRL annual metrics (valid time = fiscal period end, transaction time = filing date; restatements are kept)
 Transient  : SEC 8-K items (valid time = report date, transaction time = filing date)
Every event becomes one short memo; methods differ only in how they index and read these memos.
Output: out/qa/stream.parquet, out/qa/entities.parquet, out/qa/stream_stats.json"""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import ast, collections, datetime as dt, gzip, json, os, random, re
import pandas as pd
RAW = str(_OUT_DIR / 'raw'); OUT = str(_OUT_DIR / 'qa'); os.makedirs(OUT, exist_ok=True)
rnd = random.Random(20261105)
EX_REL = {'P169': 'chief_executive', 'P488': 'chairperson', 'P159': 'headquarters', 'P749': 'parent_organization', 'P127': 'owner', 'P1037': 'director_manager'}
EVENT = {'1.01': 'material agreement', '1.02': 'termination of a material agreement', '1.03': 'bankruptcy or receivership', '1.05': 'cybersecurity incident',
         '2.01': 'completed acquisition or disposition', '2.04': 'debt acceleration trigger', '2.05': 'exit or restructuring costs', '2.06': 'material impairment',
         '3.01': 'delisting notice', '4.01': 'auditor change', '4.02': 'non-reliance on prior financial statements', '5.01': 'change in control', '5.02': 'officer or director change'}
METRIC = {'revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'], 'total_assets': ['Assets'], 'total_liabilities': ['Liabilities'],
          'stockholders_equity': ['StockholdersEquity'], 'net_income': ['NetIncomeLoss']}
KW = re.compile(r'compan|business|enterprise|bank|corporation|manufactur|airline|conglomerate|insur|retail|automobile|holding|subsidiar|joint.stock|public limited|startup|technology|software|energy|oil|mining|media|telecom|pharma|railway|shipping|hotel|restaurant|consult|investment|asset|hedge|fund|broker|credit institution|financial', re.I)
T_EX = {
 'chief_executive': ['{e} named {v} as chief executive officer, effective {d}.', 'Effective {d}, {v} took over as CEO of {e}.', '{v} became the chief executive of {e} on {d}.'],
 'chairperson': ['{e} appointed {v} as chairperson of the board, effective {d}.', '{v} became chair of {e} on {d}.', 'Effective {d}, {v} chairs the board of {e}.'],
 'headquarters': ['{e} moved its headquarters to {v}, effective {d}.', 'Since {d}, {e} has been headquartered in {v}.', 'On {d}, {e} relocated its head office to {v}.'],
 'parent_organization': ['{e} became a subsidiary of {v} on {d}.', 'Effective {d}, the parent organization of {e} is {v}.', '{v} took {e} under its corporate group on {d}.'],
 'owner': ['Ownership of {e} passed to {v} on {d}.', 'Effective {d}, {e} is owned by {v}.', '{v} acquired ownership of {e} on {d}.'],
 'director_manager': ['{v} became managing director of {e} on {d}.', '{e} appointed {v} as director, effective {d}.'],
 'registered_name': ['SEC registrant {e} changed its registered name to {v}, effective {d}.', 'Effective {d}, the SEC registered name of {e} is {v}.'],
}
T_END = {
 'chief_executive': ['{v} stepped down as chief executive of {e} on {d}.', 'The tenure of {v} as CEO of {e} ended on {d}.'],
 'chairperson': ['{v} left the chair of {e} on {d}.', 'The term of {v} as chairperson of {e} ended on {d}.'],
 'headquarters': ['{e} ceased to be headquartered in {v} on {d}.'], 'parent_organization': ['{e} ceased to be a subsidiary of {v} on {d}.'],
 'owner': ['{v} ceased to own {e} on {d}.'], 'director_manager': ['{v} ceased to be director of {e} on {d}.'],
}
T_M = ['{e} filed its annual report on {f}: {m} for the fiscal year ended {d} was {x} USD.', 'In a filing dated {f}, {e} reported {m} of {x} USD for the fiscal year ended {d}.']
T_RS = ['{e} restated in a filing dated {f}: {m} for the fiscal year ended {d} is now {x} USD.']
T_E = ['{e} filed an 8-K on {f} reporting a {x} dated {d}.', 'Form 8-K filed by {e} on {f} discloses a {x} as of {d}.']

def parse_time(s, prec):
    """-> (date string, precision). Wikidata year-level values carry YYYY-01-01; precision 9=year 10=month 11=day."""
    if not s or s.startswith('http') or s.startswith('-'):
        return None, None
    d = s[:10]
    if d[5:7] == '00': d = d[:5] + '01' + d[7:]
    if d[8:10] == '00': d = d[:8] + '01'
    try:
        dt.date.fromisoformat(d)
    except ValueError:
        return None, None
    p = int(prec) if prec else (9 if d[5:] == '01-01' else 11)
    return d, p

def lit(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return x

rows = []; ents = {}
lab = {}
for l in open(f'{RAW}/wikidata/labels.jsonl'):
    x = json.loads(l); lab[x['q']] = x['label']
comp = {}
for l in open(f'{RAW}/wikidata/companies.jsonl'):
    x = json.loads(l)
    if x['ciks'] or x['exch'] or KW.search(x['classes'] or ''):
        comp[x['s']] = x
# one label -> one company; ambiguous labels are dropped
cnt = collections.Counter(lab[s] for s in comp if s in lab)
for s, c in comp.items():
    if s in lab and cnt[lab[s]] == 1 and not re.fullmatch(r'Q\d+', lab[s]):
        ciks = [x.lstrip('0') for x in (c['ciks'] or '').split('|') if x]
        ents[s] = {'eid': s, 'name': lab[s], 'cik': ciks[0] if ciks else None, 'country': c['country'], 'source': 'wikidata'}
print('wikidata companies', len(ents), flush=True)
sid = 0
def add(eid, rel, tau, value, vfrom, vfrom_prec, vto, vto_prec, tx, text, src, extra=None):
    global sid
    rows.append(dict(id=sid, eid=eid, rel=rel, tau=tau, value=value, vfrom=vfrom, vfrom_prec=vfrom_prec, vto=vto, vto_prec=vto_prec, tx=tx, text=text, src=src, extra=json.dumps(extra) if extra else None)); sid += 1
seen = set()
for l in open(f'{RAW}/wikidata/statements.jsonl'):
    x = json.loads(l)
    if x['s'] not in ents or x['o'] not in lab or re.fullmatch(r'Q\d+', lab[x['o']]) or x['prop'] not in EX_REL:
        continue
    a, ap = parse_time(x['start'], x.get('start_prec')); b, bp = parse_time(x.get('end'), x.get('end_prec'))
    if not a or a > '2026-10-31' or (b and b < a) or (b and b > '2026-10-31' and False):
        continue
    k = (x['s'], x['prop'], x['o'], a)
    if k in seen:
        continue
    seen.add(k); rel = EX_REL[x['prop']]; e = ents[x['s']]['name']; v = lab[x['o']]
    fmt = (lambda d, p: d[:4] if p == 9 else (d[:7] if p == 10 else d))
    add(x['s'], rel, 'Exclusive', v, a, ap, b, bp, a, rnd.choice(T_EX[rel]).format(e=e, v=v, d=fmt(a, ap)), 'wikidata', {'qid': x['o'], 'kind': 'open', 'stmt': x['st']})
    if b and b <= '2026-10-31':
        add(x['s'], rel, 'Exclusive', v, b, bp, None, None, b, rnd.choice(T_END[rel]).format(e=e, v=v, d=fmt(b, bp)), 'wikidata', {'qid': x['o'], 'kind': 'close', 'stmt': x['st']})
print('wikidata events', len(rows), flush=True)
# ---- SEC: registrants linked to Wikidata + registrants with enough filings
facts_have = set()
with gzip.open(f'{RAW}/sec/facts.jsonl.gz', 'rt') as f:
    for l in f:
        facts_have.add(int(json.loads(l)['cik']))
link = {e['cik']: k for k, e in ents.items() if e['cik']}
subs = []
with gzip.open(f'{RAW}/sec/submissions.jsonl.gz', 'rt') as f:
    for l in f:
        d = json.loads(l); d['eightk'] = lit(d['eightk']); d['formerNames'] = lit(d['formerNames']); cik = str(int(d['cik']))
        n8 = sum(1 for e in d['eightk'] if set((e['i'] or '').split(',')) & set(EVENT))
        d['n8'] = n8; d['ck'] = cik
        if cik in link or (int(cik) in facts_have and n8 >= 8 and d['tickers'] not in ('[]', [], None)):
            subs.append(d)
rnd.shuffle(subs)
subs.sort(key=lambda d: 0 if d['ck'] in link else 1)
subs = subs[:6000]
print('sec registrants', len(subs), flush=True)
sec_eid = {}
for d in subs:
    ck = d['ck']
    if ck in link:
        eid = link[ck]
    else:
        eid = 'C' + ck
        nm = re.sub(r'\s+/.*$', '', d['name']).strip()
        ents[eid] = {'eid': eid, 'name': nm, 'cik': ck, 'country': None, 'source': 'sec'}
    sec_eid[int(ck)] = eid
    e = ents[eid]['name']
    # registered names: formerNames holds [from,to) spans; the current name opens at the last `to`
    fn = sorted([x for x in (d['formerNames'] or []) if x.get('from')], key=lambda x: x['from'])
    if fn:
        tag = f'CIK {ck.zfill(10)}'
        chain = [(x['name'], x['from'][:10]) for x in fn]
        if fn[-1].get('to'):
            chain.append((d['name'], fn[-1]['to'][:10]))
        for nm, a in chain:
            add(eid, 'registered_name', 'Exclusive', re.sub(r'\s+/.*$', '', nm).strip(), a, 11, None, None, a, rnd.choice(T_EX['registered_name']).format(e=tag, v=re.sub(r'\s+/.*$', '', nm).strip(), d=a), 'sec_names', {'kind': 'open'})
    for f8 in d['eightk']:
        for it in sorted(set((f8['i'] or '').split(',')) & set(EVENT)):
            fd = f8['f']; rd = f8['r'] or fd
            if rd > fd: rd = fd
            x = EVENT[it]
            add(eid, 'material_event', 'Transient', x, rd, 11, rd, 11, fd, rnd.choice(T_E).format(e=e, f=fd, x=x, d=rd), 'sec_8k', {'item': it, 'accn': f8['a']})
print('with 8-K', len(rows), flush=True)
wanted = set(sec_eid)
with gzip.open(f'{RAW}/sec/facts.jsonl.gz', 'rt') as f:
    for l in f:
        d = json.loads(l)
        if int(d['cik']) not in wanted:
            continue
        eid = sec_eid[int(d['cik'])]; e = ents[eid]['name']
        for m, cs in METRIC.items():
            cur = {}
            for c in cs:   # priority: first concept that reports a period wins
                for r in d['facts'].get(c, []):
                    if r['u'] != 'USD' or r['val'] is None or not r.get('end') or not r.get('filed'):
                        continue
                    if r.get('start'):
                        n = (dt.date.fromisoformat(r['end']) - dt.date.fromisoformat(r['start'])).days
                        if not 330 <= n <= 400:
                            continue
                    if r['filed'] < r['end']:
                        continue
                    cur.setdefault(r['end'], {}).setdefault(c, []).append(r)
            for end, by in sorted(cur.items()):
                # walk filings in time order; inside one filing the concept priority breaks ties, so a later filing never changes what an earlier one reported
                byacc = {}
                for ci, c in enumerate(cs):
                    for r in by.get(c, []):
                        byacc.setdefault((r['filed'], r['accn']), []).append((ci, r))
                last = None
                for (filed, accn), lst in sorted(byacc.items()):
                    r = min(lst, key=lambda z: z[0])[1]
                    if r['val'] == last:
                        continue
                    x = f"{int(r['val']):,}"; mn = m.replace('_', ' '); tpl = T_M if last is None else T_RS
                    add(eid, m, 'Cumulative', x, end, 11, None, None, r['filed'], rnd.choice(tpl).format(e=e, f=r['filed'], m=mn, d=end, x=x), 'sec_xbrl', {'val': r['val'], 'restated': last is not None, 'accn': r['accn']})
                    last = r['val']
print('all', len(rows), flush=True)
df = pd.DataFrame(rows); df['_o'] = df.extra.fillna('').str.contains('"open"').astype(int); df = df.sort_values(['tx', '_o', 'id']).drop(columns='_o').reset_index(drop=True); df['id'] = range(len(df))  # at equal tx a close precedes an open
df.to_parquet(f'{OUT}/stream.parquet')
ed = pd.DataFrame(list(ents.values())); ed = ed[ed.eid.isin(df.eid.unique())]
ed.to_parquet(f'{OUT}/entities.parquet')
st = {'events': len(df), 'entities': len(ed), 'by_tau': df.tau.value_counts().to_dict(), 'by_rel': df.rel.value_counts().to_dict(), 'tx_range': [df.tx.min(), df.tx.max()], 'bi_temporal_gap_events': int((df.tx != df.vfrom).sum())}
json.dump(st, open(f'{OUT}/stream_stats.json', 'w'), indent=1); print(json.dumps(st), flush=True)

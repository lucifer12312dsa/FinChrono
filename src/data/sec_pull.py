#!/usr/bin/env python3
"""Filter EDGAR bulk zips locally to operating companies that filed a 10-K.
Writes out/raw/sec/{submissions.jsonl.gz,facts.jsonl.gz}. Bulk zips stay in TMP_DIR outside out/."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import gzip, json, os, re, subprocess, zipfile
TMP = str(_TMP_DIR / 'sec'); OUT = str(_OUT_DIR / 'raw' / 'sec'); os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
UA = 'FinChrono-academic-research (temporal KG study) contact: research-bot'
for n in ['submissions', 'companyfacts']:
    if not os.path.exists(f'{TMP}/{n}.zip'):
        subprocess.check_call(['curl', '-sSL', '--retry', '5', '-H', f'User-Agent: {UA}', '-o', f'{TMP}/{n}.tmp',
                               ('https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip' if n == 'submissions' else 'https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip')])
        os.rename(f'{TMP}/{n}.tmp', f'{TMP}/{n}.zip')
    print('have', n, os.path.getsize(f'{TMP}/{n}.zip'), flush=True)

ITEMS8K = True
def cols(rec):
    keys = ['form', 'filingDate', 'reportDate', 'items', 'accessionNumber', 'primaryDocument']
    n = len(rec.get('form', []))
    for i in range(n):
        yield {k: (rec.get(k) or [''] * n)[i] for k in keys}

z = zipfile.ZipFile(f'{TMP}/submissions.zip')
names = z.namelist(); print('submission files', len(names), flush=True)
main = [x for x in names if re.fullmatch(r'CIK\d+\.json', x)]
extras = {}
for x in names:
    m = re.fullmatch(r'(CIK\d+)-submissions-\d+\.json', x)
    if m:
        extras.setdefault(m.group(1), []).append(x)
kept = 0
with gzip.open(f'{OUT}/submissions.jsonl.gz.tmp', 'wt') as f:
    for k, x in enumerate(main):
        d = json.loads(z.read(x))
        recent = d.get('filings', {}).get('recent', {})
        forms = set(recent.get('form', []))
        if d.get('entityType') != 'operating':
            continue
        fl = list(cols(recent))
        for e in extras.get(x[:-5], []):
            fl += list(cols(json.loads(z.read(e))))
        if not any(r['form'] in ('10-K', '10-K405', '10-KSB') for r in fl):
            continue
        eight = [{'f': r['filingDate'], 'r': r['reportDate'], 'i': r['items'], 'a': r['accessionNumber'], 'd': r['primaryDocument']} for r in fl if r['form'] == '8-K']
        tenk = [{'f': r['filingDate'], 'r': r['reportDate'], 'a': r['accessionNumber']} for r in fl if r['form'] in ('10-K', '10-K405', '10-KSB')]
        f.write(json.dumps({'cik': d.get('cik'), 'name': d.get('name'), 'tickers': d.get('tickers'), 'exchanges': d.get('exchanges'), 'sic': d.get('sic'),
                            'sicDescription': d.get('sicDescription'), 'stateOfIncorporation': d.get('stateOfIncorporation'), 'formerNames': d.get('formerNames'),
                            'eightk': eight, 'tenk': tenk}) + '\n')
        kept += 1
        if k % 20000 == 0:
            print('sub', k, kept, flush=True)
os.rename(f'{OUT}/submissions.jsonl.gz.tmp', f'{OUT}/submissions.jsonl.gz')
print('kept companies', kept, flush=True)

ciks = {int(json.loads(l)['cik']) for l in gzip.open(f'{OUT}/submissions.jsonl.gz', 'rt')}
CONCEPTS = {'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'Liabilities', 'StockholdersEquity', 'Assets', 'NetIncomeLoss'}
DEI = {'EntityCommonStockSharesOutstanding', 'AuditorName', 'EntityAddressCityOrTown'}
z = zipfile.ZipFile(f'{TMP}/companyfacts.zip'); kept = 0
with gzip.open(f'{OUT}/facts.jsonl.gz.tmp', 'wt') as f:
    for k, x in enumerate(z.namelist()):
        m = re.fullmatch(r'CIK(\d+)\.json', x)
        if not m or int(m.group(1)) not in ciks:
            continue
        d = json.loads(z.read(x)); out = {}
        for tax, want in (('us-gaap', CONCEPTS), ('dei', DEI)):
            for c, body in d.get('facts', {}).get(tax, {}).items():
                if c not in want:
                    continue
                rows = []
                for unit, arr in body.get('units', {}).items():
                    for r in arr:
                        if r.get('form') in ('10-K', '10-K/A') and (tax == 'dei' or r.get('fp') == 'FY'):
                            rows.append({'u': unit, 'start': r.get('start'), 'end': r.get('end'), 'val': r.get('val'), 'fy': r.get('fy'), 'filed': r.get('filed'), 'form': r.get('form'), 'accn': r.get('accn')})
                if rows:
                    out[c] = rows
        if out:
            f.write(json.dumps({'cik': int(m.group(1)), 'facts': out}) + '\n'); kept += 1
        if k % 3000 == 0:
            print('facts', k, kept, flush=True)
os.rename(f'{OUT}/facts.jsonl.gz.tmp', f'{OUT}/facts.jsonl.gz')
open(f'{OUT}/DONE', 'w').write('ok\n'); print('DONE facts', kept, flush=True)

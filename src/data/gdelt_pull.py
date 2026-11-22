#!/usr/bin/env python3
"""GDELT 1.0 daily export -> company events 2018-01-01..2023-12-31. Keep events with an MNC-typed actor and two named actors,
dated the day the event was added (drops retrospective mentions). One gz per month-of-days locally, merged at the end."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import csv, datetime as dt, gzip, io, os, sys, time, urllib.request as u, zipfile
from concurrent.futures import ThreadPoolExecutor
csv.field_size_limit(10**7)
OUT = str(_OUT_DIR / 'raw' / 'gdelt'); DAY = str(_TMP_DIR / 'gdelt'); os.makedirs(OUT, exist_ok=True); os.makedirs(DAY, exist_ok=True)
UA = {'User-Agent': 'FinChrono-academic-research (temporal KG study) contact: research-bot'}
def one(d):
    tag = d.strftime('%Y%m%d'); p = f'{DAY}/{tag}.tsv.gz'
    if os.path.exists(p):
        return 0
    for i in range(5):
        try:
            raw = u.urlopen(u.Request(f'https://data.gdeltproject.org/events/{tag}.export.CSV.zip', headers=UA), timeout=180).read(); break
        except Exception as e:
            if '404' in repr(e):
                print('missing', tag, flush=True); gzip.open(p, 'wt').close(); return 0
            time.sleep(10 * (i + 1))
    else:
        print('failed', tag, flush=True); return 0
    z = zipfile.ZipFile(io.BytesIO(raw)); n = 0
    with gzip.open(p + '.tmp', 'wt') as f:
        for r in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), errors='ignore'), delimiter='\t'):
            if len(r) < 58 or not r[6] or not r[16] or not (r[12] == 'MNC' or r[22] == 'MNC') or r[1] != tag:
                continue
            f.write('\t'.join([r[1], r[6], r[12], r[7], r[16], r[22], r[17], r[26], r[29], r[30], r[31], r[32], r[34], r[57]]) + '\n'); n += 1
    os.rename(p + '.tmp', p)
    return n
days = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range((dt.date(2023, 12, 31) - dt.date(2018, 1, 1)).days + 1)]
tot = 0
with ThreadPoolExecutor(4) as ex:
    for k, n in enumerate(ex.map(one, days)):
        tot += n
        if k % 50 == 0:
            print('day', k, days[k], 'rows', tot, flush=True)
with gzip.open(f'{OUT}/events.tsv.gz.tmp', 'wt') as f:
    f.write('date\ta1\ta1t\ta1c\ta2\ta2t\ta2c\tcode\tquad\tgold\tment\tsrc\ttone\turl\n')
    for d in days:
        p = f"{DAY}/{d.strftime('%Y%m%d')}.tsv.gz"
        if os.path.exists(p):
            with gzip.open(p, 'rt') as g:
                f.write(g.read())
os.rename(f'{OUT}/events.tsv.gz.tmp', f'{OUT}/events.tsv.gz')
open(f'{OUT}/DONE', 'w').write('ok\n'); print('DONE', flush=True)

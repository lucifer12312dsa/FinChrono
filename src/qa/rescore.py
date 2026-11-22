#!/usr/bin/env python3
"""Recompute every metric from stored per-question rows so that all methods share one scorer. Prints one line per run."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import summarize
out = {}
pat = sys.argv[1] if len(sys.argv) > 1 else '*'
for f in sorted(glob.glob(f'{_OUT_DIR}/eval/{pat}_rows.json')):
    rows = json.load(open(f)); name = os.path.basename(f).replace('_rows.json', '')
    def group(key, keep=lambda r: True):
        g = {}
        for r in rows:
            if keep(r):
                g.setdefault(key(r), []).append(r)
        return {str(k): summarize(v) for k, v in sorted(g.items(), key=lambda x: str(x[0]))}
    res = dict(name=name, overall_main=summarize([r for r in rows if r['q']['track'] == 'main']), hist=summarize([r for r in rows if r['q']['track'] == 'hist']),
               by_family=group(lambda r: r['q']['track'] + ':' + r['q']['family']),
               by_flips=group(lambda r: min(r['q']['flips'], 4), lambda r: r['q']['family'] == 'state' and r['q']['track'] == 'main'))
    out[name] = res; o = res['overall_main']
    print(name.ljust(26), 'EM %.3f F1 %.3f SFHR %.3f FLR %.3f IoU %.3f' % (o['em'], o['f1'], o['sfhr'], o['flr'], o['iou']), {k.split(':')[1]: round(v['em'], 2) for k, v in res['by_family'].items() if k.startswith('main')}, 'hist %.3f' % res['hist'].get('em', float('nan')))
json.dump(out, open(str(_OUT_DIR / 'eval' / '_rescored.json'), 'w'), indent=1)

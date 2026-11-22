#!/usr/bin/env python3
"""Pull time-qualified company statements from Wikidata (CEO, chair, HQ, parent, owner, director).
Stage-1: all statements of a property with a start qualifier. Stage-2: keep subjects that are companies.
Output: out/raw/wikidata/{statements.jsonl,companies.jsonl}. Resumable per property / batch."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import json, os, sys, time, urllib.request as u, urllib.parse as up
OUT = str(_OUT_DIR / 'raw' / 'wikidata'); os.makedirs(OUT, exist_ok=True)
UA = {'User-Agent': 'FinChrono-academic-research (temporal KG study) contact: research-bot', 'Accept': 'application/sparql-results+json'}
PROPS = {'P169': 'chief executive officer', 'P488': 'chairperson', 'P159': 'headquarters location', 'P127': 'owned by', 'P749': 'parent organization', 'P1037': 'director / manager'}

def sparql(q, tries=6):
    for i in range(tries):
        try:
            req = u.Request('https://query.wikidata.org/sparql?format=json', data=up.urlencode({'query': q}).encode(), headers=UA)
            return json.loads(u.urlopen(req, timeout=75).read())['results']['bindings']
        except Exception as e:
            print('retry', i, repr(e)[:120], flush=True); time.sleep(20 * (i + 1))
    raise RuntimeError('sparql failed')

def v(b, k):
    return b[k]['value'] if k in b else None

stm = os.path.join(OUT, 'statements.jsonl')
# statements are fetched per batch of company ids: whole-property scans of P159/P127 exceed the 60 s query limit
uni = os.path.join(OUT, 'universe.txt')
if not os.path.exists(uni):
    ids = set()
    for name, q in {'cik': 'SELECT DISTINCT ?s WHERE { ?s wdt:P5531 ?x }', 'exch': 'SELECT DISTINCT ?s WHERE { ?s wdt:P414 ?x }',
                    'ceo': 'SELECT DISTINCT ?s WHERE { ?s p:P169 ?st . ?st pq:P580 ?x }'}.items():
        rows = sparql(q); ids |= {v(b, 's').split('/')[-1] for b in rows}; print(name, len(rows), len(ids), flush=True)
    for cls in ['Q891723', 'Q6881511', 'Q783794', 'Q4830453', 'Q22687', 'Q46970', 'Q249556']:
        try:
            rows = sparql(f'SELECT DISTINCT ?s WHERE {{ ?s wdt:P31 wd:{cls} . ?s p:P159|p:P127|p:P749|p:P488 ?st . ?st pq:P580 ?x }}', tries=2)
            ids |= {v(b, 's').split('/')[-1] for b in rows}; print(cls, len(rows), len(ids), flush=True)
        except RuntimeError:
            print('class scan failed', cls, flush=True)
    open(uni, 'w').write('\n'.join(sorted(ids)))
ids = [x for x in open(uni).read().split() if x.startswith('Q')]
ckpt = os.path.join(OUT, 'batches_done.txt')
dn = {int(x) for x in open(ckpt).read().split()} if os.path.exists(ckpt) else set()
vp = ' '.join('p:' + p for p in PROPS)
with open(stm, 'a') as f:
    for bi, i in enumerate(range(0, len(ids), 250)):
        if bi in dn:
            continue
        vals = ' '.join('wd:' + x for x in ids[i:i + 250])
        rows = sparql(f"""SELECT ?pp ?st ?s ?o ?a ?ap ?b ?bp WHERE {{ VALUES ?s {{ {vals} }} VALUES ?pp {{ {vp} }} ?s ?pp ?st . ?st pqv:P580 ?av . ?av wikibase:timeValue ?a ; wikibase:timePrecision ?ap .
             ?prop wikibase:claim ?pp ; wikibase:statementProperty ?sp . ?st ?sp ?o .
             OPTIONAL {{ ?st pqv:P582 ?bv . ?bv wikibase:timeValue ?b ; wikibase:timePrecision ?bp . }} }}""")
        for b in rows:
            f.write(json.dumps({'prop': v(b, 'pp').split('/')[-1], 'st': v(b, 'st').split('/')[-1], 's': v(b, 's').split('/')[-1], 'o': v(b, 'o').split('/')[-1],
                                'start': v(b, 'a'), 'start_prec': v(b, 'ap'), 'end': v(b, 'b'), 'end_prec': v(b, 'bp')}) + '\n')
        f.flush(); open(ckpt, 'a').write(f'{bi}\n')
        if bi % 20 == 0:
            print('batch', bi, len(ids) // 250, flush=True)
        time.sleep(0.3)

subjects = sorted({json.loads(l)['s'] for l in open(stm)})
cmp = os.path.join(OUT, 'companies.jsonl')
seen = {json.loads(l)['s'] for l in open(cmp)} if os.path.exists(cmp) else set()
todo = [s for s in subjects if s not in seen]
print('subjects', len(subjects), 'todo', len(todo), flush=True)
with open(cmp, 'a') as f:
    for i in range(0, len(todo), 300):
        ids = todo[i:i + 300]
        vals = ' '.join('wd:' + x for x in ids)
        rows = sparql(f'''SELECT ?s (GROUP_CONCAT(DISTINCT ?cl;separator="|") AS ?classes) (GROUP_CONCAT(DISTINCT ?ck;separator="|") AS ?ciks)
           (GROUP_CONCAT(DISTINCT ?ex;separator="|") AS ?exch) (SAMPLE(?cn) AS ?country) WHERE {{ VALUES ?s {{ {vals} }}
             OPTIONAL {{ ?s wdt:P31 ?c . ?c rdfs:label ?cl . FILTER(LANG(?cl)="en") }} OPTIONAL {{ ?s wdt:P5531 ?ck }}
             OPTIONAL {{ ?s wdt:P414 ?e . ?e rdfs:label ?ex . FILTER(LANG(?ex)="en") }} OPTIONAL {{ ?s wdt:P17 ?k . ?k rdfs:label ?cn . FILTER(LANG(?cn)="en") }}
           }} GROUP BY ?s''')
        for b in rows:
            f.write(json.dumps({'s': v(b, 's').split('/')[-1], 'classes': v(b, 'classes'), 'ciks': v(b, 'ciks'), 'exch': v(b, 'exch'), 'country': v(b, 'country')}) + '\n')
        f.flush()
        if i % 3000 == 0:
            print('companies', i, flush=True)
        time.sleep(0.5)
lab = os.path.join(OUT, 'labels.jsonl')
have = {json.loads(l)['q'] for l in open(lab)} if os.path.exists(lab) else set()
qs = sorted({x for l in open(stm) for x in (json.loads(l)['s'], json.loads(l)['o'])} - have)
with open(lab, 'a') as f:
    for i in range(0, len(qs), 400):
        vals = ' '.join('wd:' + x for x in qs[i:i + 400] if x.startswith('Q'))
        for b in sparql(f'SELECT ?q ?l WHERE {{ VALUES ?q {{ {vals} }} ?q rdfs:label ?l . FILTER(LANG(?l)="en") }}'):
            f.write(json.dumps({'q': v(b, 'q').split('/')[-1], 'label': v(b, 'l')}) + '\n')
        f.flush()
        if i % 4000 == 0:
            print('labels', i, len(qs), flush=True)
        time.sleep(0.3)
open(os.path.join(OUT, 'DONE'), 'w').write('ok\n')
print('DONE', flush=True)

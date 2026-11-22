#!/usr/bin/env python3
"""EM, Token-F1, SFHR, FLR, Temporal Evidence IoU on Fin-ChronoQA predictions."""
import collections, json, re, string, unicodedata
def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower().strip()
    s = re.sub(r'^\[[^\]]*\]\s*', '', s)   # a record prefix copied into the answer is not an error
    s = re.sub(r'\b(usd|us\$|\$)\b|\$', '', s)
    s = re.sub(r'[−–]', '-', s)
    s = re.sub(r'(?<=\d),(?=\d)', '', s)
    s = ''.join(c for c in s if c not in set(string.punctuation) - {'-', '>', '.'})
    s = re.sub(r'\b(the|a|an)\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().rstrip('.')
def norm_list(s):
    # arrows separate items; a comma may be part of a company name, so it separates only when no arrow is present
    t = str(s)
    parts = re.split(r'->|\u2192', t) if re.search(r'->|\u2192', t) else re.split(r';|\n|,(?=\s*[A-Z])', t)
    return [norm(x) for x in parts if norm(x)]
def em(pred, q):
    if q['family'] == 'lineage':
        return norm_list(pred) == [norm(x) for x in q['answer_list']]
    return norm(pred) == norm(q['answer'])
def f1(pred, q):
    a = norm(pred).split(); b = norm(q['answer']).split()
    if not a or not b:
        return float(a == b)
    c = collections.Counter(a) & collections.Counter(b); n = sum(c.values())
    if n == 0:
        return 0.0
    p, r = n / len(a), n / len(b)
    return 2 * p * r / (p + r)
def is_stale(pred, q):
    return norm(pred) in {norm(x) for x in q.get('stale', [])} and not em(pred, q)
def is_future(pred, q):
    fut = {norm(x) for x in q.get('future', [])}
    if q['family'] == 'lineage':
        gold = {norm(x) for x in q['answer_list']}
        return any(x in fut and x not in gold for x in norm_list(pred))
    return norm(pred) in fut and not em(pred, q)
ALIAS = {}   # record id -> canonical id: statements asserting the same value of the same (entity, relation) are one piece of evidence
def set_alias(m):
    ALIAS.clear(); ALIAS.update(m)
def iou(ids, q):
    a = {ALIAS.get(str(x).lstrip('e'), str(x).lstrip('e')) for x in (ids or [])}; b = {ALIAS.get(str(x), str(x)) for x in q['evidence']}
    return len(a & b) / max(1, len(a | b))
def summarize(rows):
    """rows: list of dict(q=question, pred=str, ids=list). Returns metric dict."""
    n = len(rows)
    if not n:
        return {}
    st = [r for r in rows if r['q'].get('stale')]; ft = [r for r in rows if r['q'].get('future')]
    return dict(n=n, em=sum(em(r['pred'], r['q']) for r in rows) / n, f1=sum(f1(r['pred'], r['q']) for r in rows) / n,
                sfhr=sum(is_stale(r['pred'], r['q']) for r in st) / max(1, len(st)), n_stale=len(st),
                flr=sum(is_future(r['pred'], r['q']) for r in ft) / max(1, len(ft)), n_future=len(ft),
                iou=sum(iou(r['ids'], r['q']) for r in rows) / n, empty=sum(not str(r['pred']).strip() for r in rows) / n)

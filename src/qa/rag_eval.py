#!/usr/bin/env python3
"""Retrieval-and-read baselines on Fin-ChronoQA. Every method sees only memos with tx <= t_q (hard snapshot filter applied uniformly); they differ in ranking and in what they keep.
 dense       : Dense-RAG, BGE-M3 cosine top-k
 timeweight  : Time-Weighted Memory, alpha*cos + (1-alpha)*exp(-lambda*age_days); (alpha, lambda) chosen on val retrieval recall
 memgpt      : MemGPT-FIFO, per-entity FIFO of the last W memos + recursive summary of evicted memos + archival top-k
 lightrag    : GraphRAG/LightRAG-style static graph retrieval: linked entity (+ neighbours for 2-hop) edges ranked by cos and node degree, timestamps only as text
 gentkg      : serialized (entity, relation) history, newest last, LoRA from train_gentkg.py
 coh         : Chain-of-History: first LLM step selects important history lines, second reads them
All generation uses the same answer contract and temperature 0."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import argparse, collections, json, math, os, re, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from metrics import summarize, em
ap = argparse.ArgumentParser(); ap.add_argument('--methods', default='dense,timeweight,memgpt,lightrag,gentkg,coh'); ap.add_argument('--split', default='test'); ap.add_argument('--k', type=int, default=10)
ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct'); ap.add_argument('--tag', default='7b'); ap.add_argument('--gentkg_lora', default=''); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--drop', type=float, default=0.0); ap.add_argument('--delay', type=float, default=0.0); ap.add_argument('--qfile', default='questions.json'); ap.add_argument('--qemb', default='questions.npy'); ap.add_argument('--gpu_util', type=float, default=0.78)
a = ap.parse_args()
ROOT = str(_OUT_DIR); OUT = f'{ROOT}/qa'; EV = f'{ROOT}/eval'; os.makedirs(EV, exist_ok=True)
df = pd.read_parquet(f'{OUT}/stream.parquet'); sid = df.id.values; pos_of = {int(x): i for i, x in enumerate(sid)}; txt = df.text.values; tx = df.tx.values; eid = df.eid.values
if a.drop or a.delay:   # same degraded stream as eval_agent.Env: lost memos become never visible, late ones appear later
    rs = np.random.RandomState(0); keep = rs.rand(len(df)) >= a.drop; tx = tx.copy()
    idx_k = np.where(keep)[0]; late = rs.rand(len(idx_k)) < a.delay; extra = rs.randint(1, 31, len(idx_k))
    ok = df.tx.values[idx_k] >= '1700-01-01'
    t2 = (pd.to_datetime(pd.Series(df.tx.values[idx_k]).where(ok), errors='coerce') + pd.to_timedelta(extra, unit='D')).dt.strftime('%Y-%m-%d').values
    tx[idx_k[late & ok]] = t2[late & ok]; tx[~keep] = '2150-01-01'   # inside the pandas date range: never visible
txd = (pd.to_datetime(pd.Series(tx), errors='coerce', format='%Y-%m-%d').fillna(pd.Timestamp('1900-01-01')) - pd.Timestamp('1900-01-01')).dt.days.values
qs_all = json.load(open(f'{OUT}/{a.qfile}')); qid2i = {q['qid']: i for i, q in enumerate(qs_all)}
sel = [q for q in qs_all if (q['split'] == a.split and q['track'] == 'main') or (a.split == 'test' and q['track'] == 'hist')]
if a.limit:
    import random as _r; _r.Random(0).shuffle(sel); sel = sel[:a.limit]
print('questions', len(sel), flush=True)
dev = 'cuda'
def qday(q): return (pd.Timestamp(q['t_q']) - pd.Timestamp('1900-01-01')).days
ents = pd.read_parquet(f'{OUT}/entities.parquet')
nm2e = {r.name.lower(): r.eid for r in ents.itertuples()}
for r in ents.itertuples():
    if isinstance(r.cik, str) and r.cik:
        nm2e['cik ' + r.cik.zfill(10)] = r.eid
names_sorted = sorted(nm2e, key=len, reverse=True)
def link(q):
    """string entity linking from the question text: longest known name that occurs in it (no gold entity id)"""
    s = q['question'].lower()
    toks = re.findall(r"[\w&'.\-]+", s); best = None
    grams = set()
    for n in range(1, 9):
        for i in range(len(toks) - n + 1):
            grams.add(' '.join(toks[i:i + n]))
    c = [g for g in grams if g in nm2e]
    return nm2e[max(c, key=len)] if c else None
ent_idx = collections.defaultdict(list)
for i, e in enumerate(eid):
    ent_idx[e].append(i)
ent_idx = {k: np.array(v) for k, v in ent_idx.items()}
order = np.arange(len(df))
vals = df.value.values; rels = df.rel.values
deg = {k: len(v) for k, v in ent_idx.items()}
nbrs = collections.defaultdict(list)      # (visible day, neighbour): a link only exists from its filing date on
for i in np.where(df.tau.values == 'Exclusive')[0]:
    v = str(vals[i]).lower()
    if v in nm2e:
        nbrs[eid[i]].append((txd[i], nm2e[v]))
def topk_dense(q, k):
    i = qid2i[q['qid']]; sc = (M @ Q[i]).float(); sc[TXD > qday(q)] = -9
    v, ix = torch.topk(sc, k); return ix.cpu().numpy()
ALPHA, LAM = 0.7, 0.002
def topk_tw(q, k, alpha=None, lam=None):
    alpha = ALPHA if alpha is None else alpha; lam = LAM if lam is None else lam
    i = qid2i[q['qid']]; age = qday(q) - TXD; sc = alpha * (M @ Q[i]).float() + (1 - alpha) * torch.exp(-lam * age.clamp(min=0)); sc[age < 0] = -9
    v, ix = torch.topk(sc, k); return ix.cpu().numpy()
def tune_tw():
    """pick (alpha, lambda) by evidence recall on val"""
    global ALPHA, LAM
    val = [q for q in qs_all if q['split'] == 'val' and q['track'] == 'main'][:300]; best = (-1, None)
    for al in (.3, .5, .7, .9):
        for la in (.0005, .002, .01, .05):
            r = np.mean([len(set(q['evidence']) & set(sid[topk_tw(q, a.k, al, la)].tolist())) / len(q['evidence']) for q in val])
            if r > best[0]: best = (r, (al, la))
    ALPHA, LAM = best[1]; print('timeweight selected', best, flush=True)
def linked_rows(q, e):
    ix = ent_idx.get(e); 
    if ix is None: return np.array([], dtype=int)
    return ix[txd[ix] <= qday(q)]
def lightrag(q, k):
    e = link(q); pool = []
    if e is None:
        return topk_dense(q, k)
    rows = linked_rows(q, e); nb = sorted({n for d_, n in nbrs.get(e, ()) if d_ <= qday(q)})[:20]
    if q['question'].lower().find(' of the ') >= 0 or True:
        for n in nb:
            pool.append(linked_rows(q, n))
    allr = np.concatenate([rows] + pool) if pool else rows
    if len(allr) == 0:
        return topk_dense(q, k)
    i = qid2i[q['qid']]; ix = torch.tensor(allr, device=dev); sc = (M[ix] @ Q[i]).float()
    vis = {x: len(linked_rows(q, x)) for x in {eid[j] for j in allr}}      # degree as of t_q
    dg = torch.tensor([math.log1p(vis[eid[j]]) for j in allr], device=dev, dtype=torch.float32)
    sc = sc + 0.02 * dg
    v, o = torch.topk(sc, min(k, len(allr))); return allr[o.cpu().numpy()]
def serial_history(q, cap=20):
    """(entity, relation) history in time order: entity linked from text; relations from question words"""
    e = link(q); rows = linked_rows(q, e) if e else np.array([], dtype=int)
    if len(rows) == 0: return rows
    i = qid2i[q['qid']]; sc = (M[torch.tensor(rows, device=dev)] @ Q[i]).float().cpu().numpy()
    top = rows[np.argsort(-sc)[:cap]]
    return top[np.argsort(tx[top], kind='stable')]
INSTR = """You answer questions about companies using the memory records below. Each record has an id in brackets. Use the records only. Reply with one JSON object:
{"answer": <final answer>, "cited_event_ids": [<ids of the records you used>]}
Lists are written "A -> B -> C" in chronological order. Amounts are written exactly as in the records with thousands separators; changes carry a sign."""
def prompt(q, ids, extra=''):
    mem = '\n'.join(f'[{sid[j]}] {txt[j]}' for j in ids)
    return [{'role': 'system', 'content': INSTR}, {'role': 'user', 'content': f"{extra}Memory records:\n{mem}\n\nQuestion: {q['question']}"}]
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
llm = LLM(a.model, enable_lora=bool(a.gentkg_lora), max_lora_rank=64, max_model_len=4096, gpu_memory_utilization=a.gpu_util, seed=0, dtype='bfloat16')
tok = llm.get_tokenizer(); sp = SamplingParams(temperature=0, max_tokens=256)
M = torch.from_numpy(np.load(f'{OUT}/emb/memos.npy').astype(np.float32)).to(dev, dtype=torch.float16)
Q = torch.from_numpy(np.load(f'{OUT}/emb/{a.qemb}').astype(np.float32)).to(dev, dtype=torch.float16)
TXD = torch.tensor(txd, device=dev, dtype=torch.float32)
def gen(msgs, lora=None):
    ps = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    return [o.outputs[0].text for o in llm.generate(ps, sp, lora_request=lora, use_tqdm=False)]
def parse(t):
    m = re.search(r'\{.*\}', t, re.S)
    if m:
        try:
            j = json.loads(m.group(0)); ans = j.get('answer', '')
            return (' -> '.join(str(x) for x in ans) if isinstance(ans, (list, tuple)) else ('' if ans is None else str(ans))), j.get('cited_event_ids') or []
        except Exception:
            mm = re.search(r'"answer"\s*:\s*("(?:[^"\\\\]|\\\\.)*"|[^,{}\[\]"]+(?:,\d{3})*)\s*,\s*"cited', t)
            if mm: return mm.group(1).strip().strip('"'), re.findall(r'\d+', t[mm.end():])
    return t.strip()[:200], []
def finish(name, rows, wall):
    def group(key):
        g = {}
        for r in rows:
            g.setdefault(key(r), []).append(r)
        return {str(k): summarize(v) for k, v in sorted(g.items(), key=lambda x: str(x[0]))}
    res = dict(name=name, overall_main=summarize([r for r in rows if r['q']['track'] == 'main']), hist=summarize([r for r in rows if r['q']['track'] == 'hist']),
               by_family=group(lambda r: r['q']['track'] + ':' + r['q']['family']), by_flips=group(lambda r: 'F' + str(min(r['q']['flips'], 4)) if r['q']['family'] == 'state' and r['q']['track'] == 'main' else 'other'), wall=wall)
    json.dump(res, open(f'{EV}/{name}_{a.split}.json', 'w'), indent=1); json.dump([{k: r[k] for k in ('q', 'pred', 'ids')} for r in rows], open(f'{EV}/{name}_{a.split}_rows.json', 'w'))
    print(name, json.dumps(res['overall_main']), json.dumps(res['hist']), flush=True)
def run(name, build, lora=None):
    fp = f'{EV}/{name}_{a.split}.json'
    if os.path.exists(fp):
        print('have', name, flush=True); return
    t0 = time.time(); msgs = [build(q) for q in sel]
    outs = gen(msgs, lora); rows = []
    for q, o in zip(sel, outs):
        p, ids = parse(o); rows.append(dict(q=q, pred=p, ids=ids))
    finish(name, rows, time.time() - t0)
meth = a.methods.split(',')
if 'dense' in meth:
    run(f'dense_{a.tag}', lambda q: prompt(q, topk_dense(q, a.k)))
if 'timeweight' in meth:
    tune_tw(); run(f'timeweight_{a.tag}', lambda q: prompt(q, topk_tw(q, a.k)))
if 'lightrag' in meth:
    run(f'lightrag_{a.tag}', lambda q: prompt(q, lightrag(q, a.k)))
if 'memgpt' in meth and not os.path.exists(f'{EV}/memgpt_{a.tag}_{a.split}.json'):
    # step 1: recursive summary of evicted memos, per question (a summary reads the 30 memos that left the FIFO window)
    t0 = time.time(); W = 8; sums = []
    sum_msgs = []
    for q in sel:
        e = link(q); rows = linked_rows(q, e) if e else np.array([], dtype=int)
        rows = rows[np.argsort(tx[rows], kind='stable')] if len(rows) else rows
        ev = rows[:-W][-30:] if len(rows) > W else np.array([], dtype=int)
        sum_msgs.append(ev)
    idx = [i for i, ev in enumerate(sum_msgs) if len(ev)]
    S = {i: '' for i in range(len(sel))}
    prompts = [[{'role': 'system', 'content': 'Summarize these company records into a compact archival note. Keep names, dates and amounts.'}, {'role': 'user', 'content': '\n'.join(txt[j] for j in sum_msgs[i])}] for i in idx]
    for i, o in zip(idx, gen(prompts)): S[i] = o
    def build(q, i=None): raise NotImplementedError
    msgs = []
    for i, q in enumerate(sel):
        e = link(q); rows = linked_rows(q, e) if e else np.array([], dtype=int)
        rows = rows[np.argsort(tx[rows], kind='stable')] if len(rows) else rows
        win = list(rows[-W:]); arch = [x for x in topk_dense(q, 5) if x not in win]
        extra = f'Archival summary: {S[i]}\n\n' if S[i] else ''
        msgs.append(prompt(q, arch + win, extra))
    rows = []
    for q, o in zip(sel, gen(msgs)):
        p, ids = parse(o); rows.append(dict(q=q, pred=p, ids=ids))
    finish(f'memgpt_{a.tag}', rows, time.time() - t0)
if 'gentkg' in meth and a.gentkg_lora:
    lr = LoRARequest('g', 1, f'{ROOT}/{a.gentkg_lora}')
    run(f'gentkg_{a.tag}', lambda q: prompt(q, serial_history(q)), lora=lr)
if 'coh' in meth and not os.path.exists(f'{EV}/coh_{a.tag}_{a.split}.json'):
    t0 = time.time(); hists = [serial_history(q, cap=30) for q in sel]
    step1 = [[{'role': 'system', 'content': 'From the numbered history lines select the ids most useful to answer the question, considering time order. Reply with a JSON list of ids.'},
              {'role': 'user', 'content': '\n'.join(f'[{sid[j]}] {txt[j]}' for j in h) + f"\n\nQuestion: {q['question']}"}] for q, h in zip(sel, hists)]
    chosen = []
    for o, h in zip(gen(step1), hists):
        ids = [pos_of[int(x)] for x in re.findall(r'\d+', o) if int(x) in pos_of]; hs = set(h.tolist()); pick = [x for x in ids if x in hs][:10]
        chosen.append(pick or h[-10:].tolist())
    rows = []
    for q, o in zip(sel, gen([prompt(q, c, 'Chain of history reasoning: read the records in time order, and decide which record holds at the question date.\n') for q, c in zip(sel, chosen)])):
        p, ids = parse(o); rows.append(dict(q=q, pred=p, ids=ids))
    finish(f'coh_{a.tag}', rows, time.time() - t0)
print('DONE', flush=True)

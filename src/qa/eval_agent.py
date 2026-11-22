#!/usr/bin/env python3
"""Evaluate an agent on Fin-ChronoQA with vLLM: interactive tool loop, temperature 0, budget B tool calls.
  eval_agent.py --name finchrono_7b_s0 --model ... --lora dir --view bitemp --split test --tracks main,hist"""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from agent_env import Env, SYSTEM, ZS_HINT, parse_json, answer_text
from metrics import summarize
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct'); ap.add_argument('--lora', default='')
ap.add_argument('--view', default='bitemp'); ap.add_argument('--split', default='test'); ap.add_argument('--tracks', default='main,hist'); ap.add_argument('--B', type=int, default=5)
ap.add_argument('--zeroshot', type=int, default=0); ap.add_argument('--abl', default=''); ap.add_argument('--drop', type=float, default=0.0); ap.add_argument('--delay', type=float, default=0.0); ap.add_argument('--qfile', default='questions.json'); ap.add_argument('--gpu_util', type=float, default=0.85); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
ROOT = str(_OUT_DIR); od = f'{ROOT}/eval'; os.makedirs(od, exist_ok=True)
fp = f'{od}/{a.name}_{a.split}.json'
if os.path.exists(fp):
    print('done', fp); sys.exit()
qs = json.load(open(f'{ROOT}/qa/{a.qfile}'))
tr = a.tracks.split(',')
sel = [q for q in qs if (q['track'] == 'main' and 'main' in tr and q['split'] == a.split) or (q['track'] == 'hist' and 'hist' in tr and a.split == 'test')]
if a.limit:
    import random as _r; _r.Random(0).shuffle(sel); sel = sel[:a.limit]
print('questions', len(sel), flush=True)
env = Env(append_only=a.view == 'append', abl=[x for x in a.abl.split(',') if x], drop=a.drop, delay=a.delay)
llm = LLM(a.model, enable_lora=bool(a.lora), max_lora_rank=64, max_model_len=4096, gpu_memory_utilization=a.gpu_util, seed=0, dtype='bfloat16')
tok = llm.get_tokenizer(); lreq = LoRARequest('a', 1, a.lora if a.lora.startswith('/') else f'{ROOT}/{a.lora}') if a.lora else None
sp = SamplingParams(temperature=0, max_tokens=400 if a.zeroshot else 400)
sysm = SYSTEM + (ZS_HINT if a.zeroshot else '')
st = [dict(q=q, msgs=[{'role': 'system', 'content': sysm}, {'role': 'user', 'content': 'Question: ' + q['question']}], done=False, pred='', ids=[], interval=None, calls=0, trace=[], invalid=0) for q in sel]
t0 = time.time(); toks = 0
for rnd in range(a.B + 1):
    act = [s for s in st if not s['done']]
    if not act:
        break
    prompts = [tok.apply_chat_template(s['msgs'], tokenize=False, add_generation_prompt=True) for s in act]
    outs = llm.generate(prompts, sp, lora_request=lreq, use_tqdm=False)
    for s, o in zip(act, outs):
        txt = o.outputs[0].text; toks += len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
        s['trace'].append(txt[:400]); j = parse_json(txt)
        if j is None or 'tool' not in j:
            s['invalid'] += 1
            if rnd == a.B or s['invalid'] >= 2:
                s['pred'] = answer_text((j or {}).get('answer', '')) if isinstance(j, dict) else ''; s['done'] = True; continue
            s['msgs'] += [{'role': 'assistant', 'content': txt}, {'role': 'user', 'content': 'Observation:\nInvalid format. Reply with one JSON tool call.'}]; continue
        if j['tool'] == 'Emit_Temporal_Verdict':
            s['pred'] = answer_text(j.get('answer', '')); s['ids'] = j.get('cited_event_ids') or []; s['interval'] = j.get('validity_interval'); s['done'] = True; continue
        if rnd == a.B:
            s['done'] = True; continue
        s['calls'] += 1; obs = env.step(j, cap=s['q']['t_q'])
        s['msgs'] += [{'role': 'assistant', 'content': txt}, {'role': 'user', 'content': 'Observation:\n' + obs}]
wall = time.time() - t0
rows = [dict(q=s['q'], pred=s['pred'], ids=s['ids'], calls=s['calls'], invalid=s['invalid'], trace=s['trace'][-3:]) for s in st]
def group(key):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return {str(k): summarize(v) for k, v in sorted(g.items(), key=lambda x: str(x[0]))}
res = dict(name=a.name, overall_main=summarize([r for r in rows if r['q']['track'] == 'main']), hist=summarize([r for r in rows if r['q']['track'] == 'hist']),
           by_family=group(lambda r: r['q']['track'] + ':' + r['q']['family']), by_flips=group(lambda r: 'F' + str(min(r['q']['flips'], 4)) if r['q']['family'] == 'state' and r['q']['track'] == 'main' else 'other'),
           wall=wall, tokens_per_q=toks / max(1, len(rows)), calls=sum(r['calls'] for r in rows) / max(1, len(rows)), args=vars(a))
json.dump(res, open(fp, 'w'), indent=1); json.dump(rows, open(fp.replace('.json', '_rows.json'), 'w'))
print(json.dumps({k: res[k] for k in ('overall_main', 'hist')}), flush=True)
print(json.dumps(res['by_family'])[:1500], flush=True)

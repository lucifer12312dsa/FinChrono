#!/usr/bin/env python3
"""Robustness set: test questions rewritten by an LLM outside the training templates. A rewrite is kept only if it still names the entity and the exact date(s) in ISO form; otherwise the original stays and is counted.
Output out/qa/questions_para.json (same qids and gold) and embeddings out/qa/emb/questions_para.npy."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import json, os, re, sys
import numpy as np, torch
from vllm import LLM, SamplingParams
OUT = str(_OUT_DIR / 'qa')
qs = json.load(open(f'{OUT}/questions.json')); sel = [q for q in qs if q['split'] == 'test']
llm = LLM('Qwen/Qwen2.5-14B-Instruct', max_model_len=2048, gpu_memory_utilization=0.85, seed=0, dtype='bfloat16'); tok = llm.get_tokenizer()
P = 'Rewrite this analyst question in different, natural wording. Keep its meaning. Keep the company name or CIK number exactly as written, and write every date in the form YYYY-MM-DD. Output only the rewritten question.\nQuestion: {q}'
sp = SamplingParams(temperature=0.7, max_tokens=120, seed=0)
pr = [tok.apply_chat_template([{'role': 'user', 'content': P.format(q=q['question'])}], tokenize=False, add_generation_prompt=True) for q in sel]
out = llm.generate(pr, sp, use_tqdm=False); kept = 0
import pandas as pd
e = pd.read_parquet(f'{OUT}/entities.parquet'); names = dict(zip(e.eid, e.name))
for q, o in zip(sel, out):
    t = o.outputs[0].text.strip().split('\n')[0].strip().strip('"')
    need = [q['t_q']] + ([q['t_start']] if q.get('t_start') else [])
    name = re.search(r'CIK \d{10}', q['question'])
    ok = all(d in t for d in need) and t.endswith(('?', '.')) and len(t) > 15
    if name:
        ok = ok and name.group(0) in t
    else:
        # the entity string must survive: longest capitalised span of the original question is compared case-insensitively
        ok = ok and names[q['eid']].lower() in t.lower()
    q['question_orig'] = q['question']
    if ok:
        q['question'] = t; kept += 1
print('kept', kept, 'of', len(sel), flush=True)
json.dump(qs, open(f'{OUT}/questions_para.json', 'w'))
for q in sel[:6]:
    print(q['question_orig'], '=>', q['question'], flush=True)
from transformers import AutoModel, AutoTokenizer
tk = AutoTokenizer.from_pretrained('BAAI/bge-m3'); m = AutoModel.from_pretrained('BAAI/bge-m3', torch_dtype=torch.float16).cuda().eval(); E = np.zeros((len(qs), 1024), dtype=np.float16)
with torch.no_grad():
    for i in range(0, len(qs), 256):
        b = tk([q['question'] for q in qs[i:i + 256]], padding=True, truncation=True, max_length=96, return_tensors='pt').to('cuda')
        E[i:i + 256] = torch.nn.functional.normalize(m(**b).last_hidden_state[:, 0], dim=-1).float().cpu().numpy().astype(np.float16)
np.save(f'{OUT}/emb/questions_para.npy', E); print('DONE', flush=True)

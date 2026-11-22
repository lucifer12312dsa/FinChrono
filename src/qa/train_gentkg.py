#!/usr/bin/env python3
"""GenTKG baseline: LoRA with standard cross-entropy on serialized entity history -> answer; train-window main questions only (as in the original few-shot setup)."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import argparse, json, os, re, sys, random, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct'); ap.add_argument('--out', default='gentkg/7b'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--bs', type=int, default=2); ap.add_argument('--epochs', type=int, default=1)
a = ap.parse_args()
ROOT = str(_OUT_DIR); OUT = f'{ROOT}/qa'; od = f'{ROOT}/{a.out}'; os.makedirs(od, exist_ok=True)
if os.path.exists(f'{od}/DONE'):
    sys.exit()
random.seed(a.seed); torch.manual_seed(a.seed)
df = pd.read_parquet(f'{OUT}/stream.parquet'); sid = df.id.values; txt = df.text.values; tx = df.tx.values
idx = {}
for i, e in enumerate(df.eid.values):
    idx.setdefault(e, []).append(i)
allq = json.load(open(f'{OUT}/questions.json')); qi = {q['qid']: i for i, q in enumerate(allq)}
held = set(json.load(open(f'{OUT}/held.json')))
qs = [q for q in allq if q['split'] == 'train' and q['track'] == 'main' and q['eid'] not in held and q.get('eid2') not in held]
Memb = np.load(f'{OUT}/emb/memos.npy', mmap_mode='r'); Qemb = np.load(f'{OUT}/emb/questions.npy')
INSTR = """You answer questions about companies using the memory records below. Each record has an id in brackets. Use the records only. Reply with one JSON object:
{"answer": <final answer>, "cited_event_ids": [<ids of the records you used>]}
Lists are written "A -> B -> C" in chronological order. Amounts are written exactly as in the records with thousands separators; changes carry a sign."""
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'right'; tok.pad_token = tok.pad_token or tok.eos_token
data = []
for q in qs:
    # same retrieval as at test time: the 20 most similar visible records of the entity, shown in time order
    rows = np.array([i for i in idx.get(q['eid'], []) if tx[i] <= q['t_q']], dtype=int)
    if len(rows):
        sc = np.asarray(Memb[rows]).astype(np.float32) @ Qemb[qi[q['qid']]].astype(np.float32)
        rows = rows[np.argsort(-sc)[:20]]; rows = rows[np.argsort(tx[rows], kind='stable')]
    mem = '\n'.join(f'[{sid[j]}] {txt[j]}' for j in rows)
    m = [{'role': 'system', 'content': INSTR}, {'role': 'user', 'content': f"Memory records:\n{mem}\n\nQuestion: {q['question']}"}]
    p = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True); y = json.dumps({'answer': q['answer'], 'cited_event_ids': q['evidence']}, ensure_ascii=False)
    pi = tok(p, add_special_tokens=False)['input_ids']; yi = tok(y + (tok.eos_token or ''), add_special_tokens=False)['input_ids']
    if len(pi) + len(yi) < 1400:
        data.append((pi + yi, [-100] * len(pi) + yi))
print('train', len(data), flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={'': 0})
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=64, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
for p in model.parameters():
    if p.requires_grad: p.data = p.data.to(torch.bfloat16)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train()
tot = len(data) * a.epochs // a.bs; sch = get_cosine_schedule_with_warmup(opt, 50, tot); st = 0
for ep in range(a.epochs):
    random.shuffle(data)
    for i in range(0, len(data) - a.bs + 1, a.bs):
        b = data[i:i + a.bs]; L = max(len(x[0]) for x in b)
        x = torch.full((a.bs, L), tok.pad_token_id); y = torch.full((a.bs, L), -100); m = torch.zeros((a.bs, L), dtype=torch.long)
        for k, (u, v) in enumerate(b):
            x[k, :len(u)] = torch.tensor(u); y[k, :len(v)] = torch.tensor(v); m[k, :len(u)] = 1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=x.cuda(), attention_mask=m.cuda(), labels=y.cuda()).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); st += 1
        if st % 50 == 0:
            print(st, tot, loss.item(), flush=True)
model.save_pretrained(f'{od}/lora'); open(f'{od}/DONE', 'w').write('ok'); print('DONE', flush=True)

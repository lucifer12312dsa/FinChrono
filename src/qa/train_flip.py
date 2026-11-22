#!/usr/bin/env python3
"""Paired time-flip training of the agent policy (LoRA, rank 64).
 L = SFT(t1) + SFT(t2) + lambda * [ max(0, delta - (log pi(y_new|G_t2) - log pi(y_old|G_t2))) + max(0, delta - (log pi(y_old|G_t1) - log pi(y_new|G_t1))) ]
 Each log pi(y|G) is the summed token log-probability of the answer string right after `"answer": "` in the verdict turn.
 Resumable: adapter + step are saved to out/<dir>/last every --save steps."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
_TMP_DIR.mkdir(parents=True, exist_ok=True)
import argparse, json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct'); ap.add_argument('--pairs', default='qa/pairs_bitemp_h.jsonl'); ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--lam', type=float, default=0.5); ap.add_argument('--delta', type=float, default=2.0)
ap.add_argument('--rank', type=int, default=64); ap.add_argument('--bs', type=int, default=4, help='pairs per step'); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--n_pairs', type=int, default=20000); ap.add_argument('--epochs', type=float, default=1.0); ap.add_argument('--max_len', type=int, default=1000); ap.add_argument('--save', type=int, default=500)
ap.add_argument('--gc', type=int, default=0); ap.add_argument('--fam', default='state:.2,metric:.1,event:.2,delta:.2,lineage:.2,twohop:.1'); ap.add_argument('--save_at', default=''); ap.add_argument('--stop_at', type=int, default=0, help='stop after this many pairs (same lr schedule); the adapter is written as lora')
a = ap.parse_args()
ROOT = str(_OUT_DIR); od = f'{ROOT}/{a.out}'; os.makedirs(od, exist_ok=True)
if os.path.exists(f'{od}/DONE'):
    print('done', od); sys.exit()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'right'
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
def tmpl(msgs, gen=False):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
def ids(s):
    return tok(s, add_special_tokens=False)['input_ids']
AK = '"answer": "'
def encode(msgs, cf_answer=None):
    """factual sequence: labels on all assistant turns. Returns ids, labels, (ans_start, ans_end) span in ids.
    cf_answer: replace the answer of the final verdict and truncate right after the closing quote (no CE, only for log pi)."""
    last = msgs[-1]['content']; i = last.index(AK) + len(AK); j = last.index('", "cited_event_ids"')
    prefix_ids = ids(tmpl(msgs[:-1], gen=True))
    if cf_answer is not None:
        seq = prefix_ids + ids(last[:i]); s0 = len(seq)
        ans = ids(json.dumps(cf_answer, ensure_ascii=False)[1:-1] + '",'); seq = seq + ans   # same closing token as the factual verdict
        return seq, None, (s0, len(seq))
    full = ids(tmpl(msgs))
    labels = [-100] * len(full)
    for k, m in enumerate(msgs):
        if m['role'] != 'assistant':
            continue
        p = ids(tmpl(msgs[:k], gen=True)); q = ids(tmpl(msgs[:k + 1]))
        for x in range(len(p), min(len(q), len(full))):
            labels[x] = full[x]
    s0 = len(prefix_ids) + len(ids(last[:i])); s1 = s0 + len(ids(json.dumps(json.loads(last)['answer'], ensure_ascii=False)[1:-1] + '",'))
    return full, labels, (s0, s1)
pairs = [json.loads(l) for l in open(f'{ROOT}/{a.pairs}')]
random.shuffle(pairs)
# training budget is spent where the base model violates the margin (share measured on train pairs): event, delta, lineage
W = dict(x.split(':') for x in a.fam.split(','))
byf = {}
for p in pairs:
    byf.setdefault(p['family'], []).append(p)
pairs = [p for f, v in byf.items() for p in v[:int(a.n_pairs * float(W.get(f, 0)))]]
random.shuffle(pairs)
print('pairs', len(pairs), flush=True)
t0 = time.time()
import hashlib, pickle
_ck = str(_TMP_DIR / 'enc_') + hashlib.md5(f'v2|{a.model}|{a.pairs}|{a.n_pairs}|{a.max_len}|{a.seed}|{a.fam}'.encode()).hexdigest() + '.pkl'
data = pickle.load(open(_ck, 'rb')) if os.path.exists(_ck) else []
for p in ([] if data else pairs):
    try:
        f1 = encode(p['m1']); f2 = encode(p['m2'])
        if len(f1[0]) > a.max_len or len(f2[0]) > a.max_len:
            continue
        data.append((f1, f2, encode(p['m1'], cf_answer=p['y_new']), encode(p['m2'], cf_answer=p['y_old'])))
    except (ValueError, KeyError):
        continue
if not os.path.exists(_ck):
    pickle.dump(data, open(_ck, 'wb'))
print('encoded', len(data), 'sec', time.time() - t0, 'mean len', np.mean([len(d[0][0]) for d in data]), flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
if a.gc:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads()
last = f'{od}/last'; step0 = 0
if os.path.exists(f'{last}/step.json'):
    model = PeftModel.from_pretrained(model, last, is_trainable=True); step0 = json.load(open(f'{last}/step.json'))['step']; print('resume', step0, flush=True)
else:
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
model.print_trainable_parameters()
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.to(torch.bfloat16)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
total = int(len(data) * a.epochs / a.bs); sch = get_cosine_schedule_with_warmup(opt, min(200, total // 20), total)
for _ in range(step0):
    sch.step()
if os.path.exists(f'{last}/opt.pt'):
    opt.load_state_dict(torch.load(f'{last}/opt.pt', map_location='cuda'))
def forward(seqs):
    L = max(len(s[0]) for s in seqs)
    x = torch.full((len(seqs), L), tok.pad_token_id); m = torch.zeros((len(seqs), L), dtype=torch.long); y = torch.full((len(seqs), L), -100)
    for i, s in enumerate(seqs):
        x[i, :len(s[0])] = torch.tensor(s[0]); m[i, :len(s[0])] = 1
        if s[1] is not None:
            y[i, :len(s[0])] = torch.tensor(s[1])
    x, m, y = x.cuda(), m.cuda(), y.cuda()
    # run the lm_head only on positions that carry a label or an answer span: a full vocabulary tensor per token does not fit in memory
    want = torch.zeros_like(x[:, 1:], dtype=torch.bool); want |= y[:, 1:] != -100
    for i, s in enumerate(seqs):
        want[i, s[2][0] - 1:s[2][1] - 1] = True
    core = model.base_model.model
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = core.model(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state[:, :-1]
        logits = core.lm_head(h[want])
    lp = F.log_softmax(logits.float(), -1)
    pos = torch.full_like(want, -1, dtype=torch.long); pos[want] = torch.arange(want.sum(), device=want.device)
    tgt = y[:, 1:]; lab = tgt != -100
    per = torch.zeros(x.shape[0], device=x.device); den = lab.sum(1).clamp(min=1)
    pl = -lp[pos[lab], tgt[lab]]
    per = per.index_add(0, lab.nonzero()[:, 0], pl)
    ce = per / den
    ans = []
    for i, s in enumerate(seqs):
        a0, a1 = s[2]; tk = x[i, a0:a1]
        ans.append(lp[pos[i, a0 - 1:a1 - 1], tk].sum())
    return ce, torch.stack(ans)
@torch.no_grad()
def prefix_cache(ids_):
    core = model.base_model.model
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return core.model(input_ids=torch.tensor([ids_]).cuda(), use_cache=True)
def cf_logprob(seq):
    """log pi(y_false | G): the context up to the last token before the answer is encoded without gradient; that token and the answer are differentiated."""
    ids_, _, (a0, a1) = seq
    core = model.base_model.model
    core.eval()      # checkpointing is only active in train mode and would drop the cache, i.e. the context of this pass
    out = prefix_cache(ids_[:a0 - 1])
    x = torch.tensor([ids_[a0 - 1:a1]]).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = core.model(input_ids=x, past_key_values=out.past_key_values, use_cache=True).last_hidden_state
        lg = core.lm_head(h[:, :-1])
    core.train()
    return F.log_softmax(lg.float(), -1)[0].gather(-1, x[0, 1:].unsqueeze(-1)).sum()
rng = random.Random(a.seed)
order = []
while len(order) < total * a.bs + a.bs:
    o = list(range(len(data))); rng.shuffle(o); order += o
model.train(); t0 = time.time(); agg = np.zeros(2)
for st in range(step0, total):
    batch = [data[i] for i in order[st * a.bs:(st + 1) * a.bs]]
    seqs = [b[0] for b in batch] + [b[1] for b in batch]
    ce, lp_true = forward(seqs)
    l_sft = ce.mean() * 2
    n = len(batch)
    if a.lam > 0:
        lp_cf = torch.stack([cf_logprob(b[2]) for b in batch] + [cf_logprob(b[3]) for b in batch])
        if st - step0 < 2:
            print('DBG true', [round(v, 2) for v in lp_true.tolist()], 'cf', [round(v, 2) for v in lp_cf.tolist()], 'ans', [tok.decode(b[0][0][b[0][2][0]:b[0][2][1]])[:30] for b in batch], 'cf', [tok.decode(b[2][0][b[2][2][0]:b[2][2][1]])[:30] for b in batch], flush=True)
        m1 = F.relu(a.delta - (lp_true[:n] - lp_cf[:n])); m2 = F.relu(a.delta - (lp_true[n:] - lp_cf[n:]))
        l_m = (m1 + m2).mean()
    else:
        l_m = torch.tensor(0.0)
    loss = l_sft + a.lam * l_m
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    agg += [l_sft.item(), float(l_m.detach())]
    if (st + 1) % 50 == 0:
        print(json.dumps({'step': st + 1, 'total': total, 'sft': agg[0] / 50, 'margin': agg[1] / 50, 'sec': time.time() - t0, 'mem': torch.cuda.max_memory_allocated() / 2**30}), flush=True); agg[:] = 0
    if (st + 1) % a.save == 0 or st + 1 == total:
        model.save_pretrained(last); torch.save(opt.state_dict(), f'{last}/opt.pt'); json.dump({'step': st + 1}, open(f'{last}/step.json', 'w'))
    if a.stop_at and (st + 1) * a.bs >= a.stop_at:
        break
    if str((st + 1) * a.bs) in a.save_at.split(','):
        model.save_pretrained(f"{od}/lora_{(st + 1) * a.bs}")
model.save_pretrained(f'{od}/lora')
open(f'{od}/DONE', 'w').write('ok'); print('DONE', flush=True)

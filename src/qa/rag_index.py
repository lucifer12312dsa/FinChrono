#!/usr/bin/env python3
"""BGE-M3 dense embeddings of all stream memos and questions (GPU VM). Memos go to out/qa/emb/memos.npy (Drive), questions to out/qa/emb/questions.npy."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
import json, os, time
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer
OUT = str(_OUT_DIR / 'qa'); E = f'{OUT}/emb'; os.makedirs(E, exist_ok=True)
tok = AutoTokenizer.from_pretrained('BAAI/bge-m3'); m = AutoModel.from_pretrained('BAAI/bge-m3', torch_dtype=torch.float16).cuda().eval()
@torch.no_grad()
def enc(texts, bs=512):
    idx = np.argsort([len(t) for t in texts]); res = np.zeros((len(texts), 1024), dtype=np.float16); t0 = time.time()
    for i in range(0, len(texts), bs):
        j = idx[i:i + bs]; b = tok([texts[k] for k in j], padding=True, truncation=True, max_length=96, return_tensors='pt').to('cuda')
        v = torch.nn.functional.normalize(m(**b).last_hidden_state[:, 0], dim=-1); res[j] = v.float().cpu().numpy().astype(np.float16)
        if i % (bs * 100) == 0:
            print(i, len(texts), time.time() - t0, flush=True)
    return res
df = pd.read_parquet(f'{OUT}/stream.parquet')
if not os.path.exists(f'{E}/memos.npy'):
    np.save(f'{E}/memos.npy', enc(df.text.tolist()))
qs = json.load(open(f'{OUT}/questions.json'))
np.save(f'{E}/questions.npy', enc([q['question'] for q in qs], bs=256))
json.dump([q['qid'] for q in qs], open(f'{E}/question_ids.json', 'w'))
open(f'{E}/DONE', 'w').write('ok'); print('DONE', flush=True)

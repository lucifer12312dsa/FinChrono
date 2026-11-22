#!/usr/bin/env python3
"""Collect results (from FINCHRONO_OUT) into LaTeX tables, figures and one small results.json. Seed 0 everywhere. All QA numbers are rescored from per-question rows."""
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
from paths import OUT_DIR as _OUT_DIR, TMP_DIR as _TMP_DIR, BASE_DIR as _BASE_DIR
_TMP_DIR.mkdir(parents=True, exist_ok=True)
import glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, _os.path.join(_SRC_DIR, 'qa'))
from metrics import summarize, set_alias
import pandas as pd
_st = pd.read_parquet(str(_OUT_DIR / 'qa' / 'stream.parquet'), columns=['id', 'eid', 'rel', 'tau', 'value', 'vfrom'])
# statements of one state episode are one piece of evidence; a later return to the same value (A, B, A) is a different episode
_ex = _st[_st.tau == 'Exclusive'].sort_values(['eid', 'rel', 'vfrom', 'id'], kind='stable').copy()
_ex['_ep'] = (_ex.value != _ex.groupby(['eid', 'rel']).value.shift()).groupby([_ex.eid, _ex.rel]).cumsum()
_canon = _ex.groupby(['eid', 'rel', '_ep']).id.transform('min')
set_alias({str(i): str(c) for i, c in zip(_ex.id.values, _canon.values)})
ROOT = str(_OUT_DIR); RD = f'{ROOT}/results'; os.makedirs(f'{RD}/tables', exist_ok=True); os.makedirs(f'{RD}/figs', exist_ok=True)
R = {}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C = {'ours': '#2a78d6', 'b1': '#eb6834', 'b2': '#1baf7a', 'b3': '#eda100', 'b4': '#e87ba4'}
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.linewidth': .6, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5, 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42})
def bold(vals, higher=True):
    """format with the best value of a column in bold"""
    v = [x for x in vals if x is not None]
    best = max(v) if higher else min(v)
    return [('--' if x is None else (r'\textbf{%.3f}' % x if abs(x - best) < 5e-4 else '%.3f' % x)) for x in vals]
def write(n, s):
    open(f'{RD}/tables/{n}.tex', 'w').write(s)
# ---------------- TKG
def parse_dgl(name):
    p = f'{ROOT}/baselines/logs/{name}.log'
    if not os.path.exists(p):
        return None
    t = open(p, errors='ignore').read()
    mrr = re.findall(r'MRR \(filter_ent\): ([\d.]+)', t); h = {k: re.findall(rf'Hits \(filter_ent\) @ {k}: ([\d.]+)', t) for k in (1, 3, 10)}
    if not mrr or ('hismatch' in name and 'FINAL_MARK' not in t):   # a run still in training only has validation lines
        return None
    return {'val': max(float(x) for x in mrr[:-1]) if len(mrr) > 1 else None, 'mrr': float(mrr[-1]), 'h1': float(h[1][-1]), 'h3': float(h[3][-1]), 'h10': float(h[10][-1])}
def parse_his(name):
    p = f'{ROOT}/baselines/logs/{name}.log'
    if not os.path.exists(p):
        return None
    t = open(p, errors='ignore').read()
    m = re.findall(r'(?:MRR|mrr)[^\d\n]*\(?\w*\)?[^\d\n]*([\d.]+)', t); return parse_dgl(name)
def parse_tlogic(d):
    p = f'{ROOT}/baselines/logs/tlogic_{d}_s0.log'
    if not os.path.exists(p):
        return None
    t = open(p, errors='ignore').read(); i = t.rfind('EVAL ')
    seg = t[i:] if i >= 0 else ''
    if 'test' not in seg[:300] and '_test' not in seg[:300]:
        pass
    out = {}
    for k, pat in (('h1', r'Hits@1:\s*([\d.]+)'), ('h3', r'Hits@3:\s*([\d.]+)'), ('h10', r'Hits@10:\s*([\d.]+)'), ('mrr', r'MRR:\s*([\d.]+)')):
        m = re.findall(pat, t)
        if not m:
            return None
        out[k] = float(m[-1])
    return out
def parse_kgt(d):
    p = f'{ROOT}/baselines/logs/kgt_{d}_s0.log'
    if not os.path.exists(p):
        return None
    t = open(p, errors='ignore').read(); m = re.findall(r'Test loss[^\n]*MRR=([\d.]+), Rec@1=([\d.]+), Rec@3=([\d.]+), Rec@10=([\d.]+)', t)
    if not m:
        return None
    x = [float(v) for v in m[-1]]; val = [float(v) for v in re.findall(r'Validation loss[^\n]*MRR=([\d.]+)', t)]
    return dict(mrr=x[0], h1=x[1], h3=x[2], h10=x[3], val=max(val))
def _logcl_single(d, h, sp):
    p = f'{ROOT}/baselines/logs/logcl_{d}_h{h}_{sp}_single.log'
    if not os.path.exists(p):
        return None
    t = open(p, errors='ignore').read(); m = re.findall(r'FC_SINGLE (?:for|inv) n (\d+) official_mrr ([\d.]+) single_mrr ([\d.]+) single_h1 ([\d.]+) single_h3 ([\d.]+) single_h10 ([\d.]+)', t)
    if len(m) != 2:
        return None
    a = [[float(v) for v in r] for r in m]; mean = lambda i: (a[0][i] + a[1][i]) / 2
    return dict(n=int(a[0][0] + a[1][0]), official_sample=mean(1), mrr=mean(2), h1=mean(3), h3=mean(4), h10=mean(5))
def parse_logcl(d, h):
    """per-query scores; official (snapshot-pooled) scores kept apart for the text. History length is chosen by the per-query validation score."""
    v, x = _logcl_single(d, h, 'val'), _logcl_single(d, h, 'test')
    p = f'{ROOT}/baselines/logs/logcl_{d}_h{h}_s0.log'
    if not (v and x and os.path.exists(p)):
        return None
    f = re.findall(r'\(all_filter\) MRR, Hits@ \(1,3,5\):([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)', open(p, errors='ignore').read())
    return dict(x, val=v['mrr'], hist=h, n_test=x['n'], val_official_sample=v['official_sample'], official=dict(zip(('mrr', 'h1', 'h3', 'h10'), [float(c) for c in f[-1]])))
def hawkes(d, t='final'):
    p = f'{ROOT}/hawkes/{d}/{t}_s0/result.json'
    if not os.path.exists(p):
        return None
    r = json.load(open(p)); x = dict(r['test']); x['val'] = r['val']['mrr']; return x
T = {}
for d in ('findkg', 'gdeltfin'):
    cand = {'RE-GCN': [parse_dgl(f'regcn_{d}_h{h}_s0') for h in (3, 9)], 'TiRGN': [parse_dgl(f'tirgn_{d}_h3_s0')], 'HisMatch': [parse_dgl(f'hismatch_{d}_h5_s0')], 'TLogic': [parse_tlogic(d)], 'KGTransformer': [parse_kgt(d)], 'LogCL': [parse_logcl(d, h) for h in (3, 7)], 'FinChrono': [hawkes(d, 'rule')]}
    T[d] = {}
    for k, v in cand.items():
        v = [x for x in v if x]
        T[d][k] = max(v, key=lambda x: x.get('val') or 0) if v else None
R['tkg'] = T
order = ['RE-GCN', 'TiRGN', 'HisMatch', 'TLogic', 'KGTransformer', 'LogCL', 'FinChrono']
cols = {d: {k: bold([T[d][m][k] if T[d][m] else None for m in order]) for k in ('mrr', 'h1', 'h3', 'h10')} for d in T}
s = r'\begin{tabular}{l cccc cccc}\toprule' + '\n' + r' & \multicolumn{4}{c}{FinDKG} & \multicolumn{4}{c}{GDELT-Fin}\\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}' + '\n' + r'Method & MRR & H@1 & H@3 & H@10 & MRR & H@1 & H@3 & H@10\\\midrule' + '\n'
for i, m in enumerate(order):
    s += (r'\midrule ' if m == 'FinChrono' else '') + m + ' & ' + ' & '.join(cols[d][k][i] for d in ('findkg', 'gdeltfin') for k in ('mrr', 'h1', 'h3', 'h10')) + r'\\' + '\n'
write('tkg', s + r'\bottomrule\end{tabular}' + '\n')
# ---------------- QA
def rows(n, split='test'):
    p = f'{ROOT}/eval/{n}_{split}_rows.json'
    return json.load(open(p)) if os.path.exists(p) else None
def qa(n):
    r = rows(n)
    if r is None:
        return None
    m = summarize([x for x in r if x['q']['track'] == 'main']); h = summarize([x for x in r if x['q']['track'] == 'hist'])
    fam = {}
    for f in ('state', 'lineage', 'metric', 'delta', 'event', 'twohop'):
        fam[f] = summarize([x for x in r if x['q']['track'] == 'main' and x['q']['family'] == f])
    flips = {b: summarize([x for x in r if x['q']['track'] == 'main' and x['q']['family'] == 'state' and min(x['q']['flips'], 4) == b]) for b in range(5)}
    return dict(main=m, hist=h, fam=fam, flips=flips)
QA = [('Dense-RAG', 'dense_7b'), ('Time-Weighted', 'timeweight_7b'), ('MemGPT-FIFO', 'memgpt_7b'), ('GraphRAG/LightRAG', 'lightrag_7b'), ('GenTKG', 'gentkg_7b'), ('CoH', 'coh_7b'), ('ReAct-Temporal', 'zs7'), ('FinChrono (Qwen-7B)', 'm7_s0')]
EXTRA = [('FinChrono (Llama-8B)', 'l8_s0'), ('FinChrono (Qwen-14B)', 'q14_s0')]
Q = {n: qa(k) for n, k in QA + EXTRA}; R['qa'] = {k: {a: b for a, b in v.items()} for k, v in Q.items() if v}
names = [n for n, _ in QA]
g = lambda n, f, *kk: (Q[n][f][kk[0]] if Q.get(n) and Q[n][f] else None)
col = {'em': bold([g(n, 'main', 'em') for n in names]), 'f1': bold([g(n, 'main', 'f1') for n in names]), 'sf': bold([g(n, 'main', 'sfhr') for n in names], False), 'fl': bold([g(n, 'main', 'flr') for n in names], False), 'io': bold([g(n, 'main', 'iou') for n in names]), 'h': bold([g(n, 'hist', 'em') for n in names]), 'hf': bold([g(n, 'hist', 'flr') for n in names], False)}
s = r'\begin{tabular}{l ccccc cc}\toprule' + '\n' + r' & \multicolumn{5}{c}{Test window} & \multicolumn{2}{c}{Historical track}\\ \cmidrule(lr){2-6}\cmidrule(lr){7-8}' + '\n' + r'Method & EM & F1 & SFHR$\downarrow$ & FLR$\downarrow$ & IoU & EM & FLR$\downarrow$\\\midrule' + '\n'
for i, n in enumerate(names):
    s += (r'\midrule ' if n.startswith('FinChrono') else '') + n + ' & ' + ' & '.join(col[k][i] for k in ('em', 'f1', 'sf', 'fl', 'io', 'h', 'hf')) + r'\\' + '\n'
for n, _ in EXTRA:
    if Q.get(n):
        s += n + ' & ' + ' & '.join('%.3f' % g(n, 'main', k) for k in ('em', 'f1', 'sfhr', 'flr', 'iou')) + ' & ' + '%.3f & %.3f' % (g(n, 'hist', 'em'), g(n, 'hist', 'flr')) + r'\\' + '\n'
write('qa', s + r'\bottomrule\end{tabular}' + '\n')
FAM = [('state', 'State'), ('lineage', 'Lineage'), ('metric', 'Metric'), ('delta', 'Change'), ('event', 'Event'), ('twohop', '2-hop')]
s = r'\begin{tabular}{l cccccc}\toprule' + '\n' + 'Method & ' + ' & '.join(b for _, b in FAM) + r'\\\midrule' + '\n'
cf = {f: bold([Q[n]['fam'][f]['em'] if Q.get(n) else None for n in names]) for f, _ in FAM}
for i, n in enumerate(names):
    s += (r'\midrule ' if n.startswith('FinChrono') else '') + n + ' & ' + ' & '.join(cf[f][i] for f, _ in FAM) + r'\\' + '\n'
write('qa_family', s + r'\bottomrule\end{tabular}' + '\n')
# ---------------- ablation: every row is seed 0
ABL = [('Full FinChrono', 'm7_s0'), (r'w/o Bi-Temporal State Machine', 'a7_s0'), (r'w/o Rule Excitation', 'norule'), (r'w/o Hawkes Mutual-Excitation', 'nomutual7'), (r'w/o Learnable Decay', 'nodecay7'), (r'w/o Counterfactual Margin', 'nm7_s0')]
def shock(t):
    p = f'{ROOT}/shock/{t}_s0/result.json'
    return json.load(open(p)) if os.path.exists(p) else None
sf = {'m7_s0': 'g2', 'nomutual7': 'g2_nomutual', 'nodecay7': 'g2_nodecay', 'nogate': 'g2_nogate'}; hk = {'m7_s0': 'rule', 'norule': 'final', 'nomutual7': 'nomutual', 'nodecay7': 'rule_nodecay'}
s = r'\begin{tabular}{l cccc cc}\toprule' + '\n' + r' & \multicolumn{4}{c}{Fin-ChronoQA} & \multicolumn{2}{c}{MRR}\\ \cmidrule(lr){2-5}\cmidrule(lr){6-7}' + '\n' + r'Variant & EM & Event EM & SFHR$\downarrow$ & FLR$\downarrow$ & FinDKG & GDELT-Fin\\\midrule' + '\n'
R['abl'] = {}
for n, k in ABL:
    q = qa(k); a = q['main'] if q else None; sh = shock(sf[k]) if k in sf else None; hf = hawkes('findkg', hk[k]) if k in hk else None; hg = hawkes('gdeltfin', hk[k]) if k in hk else None
    R['abl'][k] = dict(em=a and a['em'], ev=q and q['fam']['event']['em'], sfhr=a and a['sfhr'], flr=a and a['flr'], f=hf and hf['mrr'], g=hg and hg['mrr'], s=sh and sh['finchrono']['auc_pr'])
    fm = lambda x: '--' if x is None else '%.3f' % x; x = R['abl'][k]
    s += n + ' & ' + ' & '.join(fm(x[c]) for c in ('em', 'ev', 'sfhr', 'flr', 'f', 'g')) + r'\\' + '\n'
write('abl', s + r'\bottomrule\end{tabular}' + '\n')
# ---------------- shock
r = shock('g2'); r['gbdt'] = shock('gbdtg')['gbdt']; R['shock'] = {k: r[k] for k in ('finchrono', 'degree', 'recent_neg', 'ppr', 'ppr_decay', 'gbdt')}; R['shock_budget'] = {k: v['R@50'] for k, v in r['budget'].items()}
SN = [('Degree', 'degree'), ('RecentNeg', 'recent_neg'), ('PPR', 'ppr'), ('Decay-PPR', 'ppr_decay'), ('GBDT', 'gbdt'), ('FinChrono', 'finchrono')]
KS = ['auc_pr', 'R@20', 'R@50', 'R@100', 'auc_pr_h2', 'auc_pr_h3', 'R@50_h2', 'R@50_h3']
cs = {k: bold([r[b][k] for _, b in SN]) for k in KS}
s = r'\begin{tabular}{l cccc cc cc}\toprule' + '\n' + r' & \multicolumn{4}{c}{All candidates} & \multicolumn{2}{c}{AUC-PR by hop} & \multicolumn{2}{c}{R@50 by hop}\\ \cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}' + '\n' + r'Method & AUC-PR & R@20 & R@50 & R@100 & 2 & 3 & 2 & 3\\\midrule' + '\n'
for i, (n, b) in enumerate(SN):
    s += (r'\midrule ' if n == 'FinChrono' else '') + n + ' & ' + ' & '.join(cs[k][i] for k in KS) + r'\\' + '\n'
write('shock', s + r'\bottomrule\end{tabular}' + '\n')
# ---------------- robustness (main track; hist questions were not rewritten)
RB = [('Clean stream', ('m7_s0', 'zs7', 'lightrag_7b')), ('Rewritten questions', ('para_m7', 'para_zs', 'lightrag_para')), ('10\\% of events lost', ('drop0.1_m7', 'drop0.1_zs', 'lightrag_drop0.1')), ('20\\% of events lost', ('drop0.2_m7', 'drop0.2_zs', 'lightrag_drop0.2')), ('30\\% filed up to 30 days late', ('late_m7', 'late_zs', 'lightrag_late'))]
s = r'\begin{tabular}{l ccc ccc}\toprule' + '\n' + r' & \multicolumn{3}{c}{EM} & \multicolumn{3}{c}{SFHR$\downarrow$}\\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\n' + r'Condition & FinChrono & ReAct & LightRAG & FinChrono & ReAct & LightRAG\\\midrule' + '\n'; R['robust'] = {}
for n, ks in RB:
    qs_ = [qa(k) for k in ks]
    R['robust'][n] = {k: (q['main'] if q else None) for k, q in zip(ks, qs_)}
    ems = bold([q['main']['em'] if q else None for q in qs_]); sf_ = bold([q['main']['sfhr'] if q else None for q in qs_], False)
    s += n + ' & ' + ' & '.join(ems + sf_) + r'\\' + '\n'
write('robust', s + r'\bottomrule\end{tabular}' + '\n')
# ---------------- store
if os.path.exists(f'{ROOT}/eval/store_bench.json'):
    R['store'] = json.load(open(f'{ROOT}/eval/store_bench.json'))
# ---------------- datasets
meta = {d: json.load(open(f'{ROOT}/tkg/{d}/meta.json')) for d in ('findkg', 'gdeltfin')}; R['meta'] = meta; R['stream'] = json.load(open(f'{ROOT}/qa/stream_stats.json'))
# ---------------- figures: flip strata
fig, ax = plt.subplots(1, 2, figsize=(4.9, 1.75))
FN = [('FinChrono (Qwen-7B)', 'FinChrono', C['ours'], 'o'), ('ReAct-Temporal', 'ReAct', C['b1'], 's'), ('GraphRAG/LightRAG', 'LightRAG', C['b2'], '^'), ('MemGPT-FIFO', 'MemGPT', C['b3'], 'D'), ('Dense-RAG', 'Dense', C['b4'], 'v')]
xs = np.arange(5)
for n, lab, c, mk in FN:
    if Q.get(n):
        ax[0].plot(xs, [Q[n]['flips'][b]['em'] for b in range(5)], color=c, marker=mk, ms=4, lw=1.4, label=lab)
        ax[1].plot(xs[1:], [Q[n]['flips'][b]['sfhr'] for b in range(1, 5)], color=c, marker=mk, ms=4, lw=1.4, label=lab)
ax[0].set_ylabel('EM'); ax[1].set_ylabel('SFHR'); ax[0].set_ylim(0, 1.03)
for a in ax:
    a.set_xticks(xs); a.set_xticklabels(['0', '1', '2', '3', r'$\geq$4']); a.set_xlabel('State flips up to $t_q$')
ax[1].set_xlim(.7, 4.3)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=5, fontsize=7, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(.5, 1.06))
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(f'{RD}/figs/fig_flips.pdf', bbox_inches='tight'); plt.close(fig)
# ---------------- figure: budget
fig, a = plt.subplots(figsize=(2.35, 1.6)); B = [1, 2, 3, 4, 5]
for k, lab, c, mk in (('FinChrono', 'FinChrono', C['ours'], 'o'), ('DecayBFS', 'Decay BFS', C['b1'], 's'), ('UniformBFS', 'Uniform BFS', C['b2'], '^')):
    a.plot(B, [R['shock_budget'][f'{k}_B{b}'] for b in B], color=c, marker=mk, ms=4, lw=1.4, label=lab)
a.set_xlabel('Query budget $B$'); a.set_ylabel('Recall@50'); a.set_xticks(B); a.set_ylim(0, None); a.legend(fontsize=7, loc='lower right', handlelength=1.6)
fig.tight_layout(); fig.savefig(f'{RD}/figs/fig_budget.pdf', bbox_inches='tight'); plt.close(fig)
# ---------------- figure: dynamics on FinDKG
import torch
dy = torch.load(f'{ROOT}/hawkes/findkg/rule_s0/dynamics.pt', map_location='cpu'); rel = [l.split('\t')[0] for l in open(f'{ROOT}/tkg/findkg/relation2id.txt')]; n = len(rel)
hl = (np.log(2) / dy['beta'].numpy())[:n]; kp = dy['kappa'].numpy()[:n, :n]   # signed: excitation (+) or inhibition (-)
R['half_life_weeks'] = dict(zip(rel, hl.tolist()))
o = np.argsort(hl); fig, ax = plt.subplots(1, 2, figsize=(4.9, 2.2), gridspec_kw={'width_ratios': [1, 1.05], 'wspace': .55})
ax[0].barh(np.arange(n), hl[o], color=C['ours'], height=.62); ax[0].set_yticks(np.arange(n)); ax[0].set_yticklabels([rel[i].replace('_', ' ') for i in o], fontsize=6); ax[0].set_xlabel(r'Half-life $\ln 2/\beta_r$ (weeks)'); ax[0].grid(axis='y', visible=False)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('d', ['#b5471e', '#eb6834', '#f0efec', '#2a78d6', '#104281'])
vm = float(np.abs(kp).max())
im = ax[1].imshow(kp[np.ix_(o, o)], cmap=cm, aspect='equal', vmin=-vm, vmax=vm); ax[1].grid(False); ax[1].set_xticks(np.arange(n)); ax[1].set_yticks(np.arange(n)); ax[1].set_xticklabels(np.arange(1, n + 1), fontsize=5); ax[1].set_yticklabels(np.arange(1, n + 1), fontsize=5)
ax[1].set_xlabel(r'Excited relation $r_i$'); ax[1].set_ylabel(r'Exciting relation $r_j$'); fig.colorbar(im, ax=ax[1], fraction=.046, pad=.03).ax.tick_params(labelsize=5, length=2)
ax[1].text(1.0, 1.03, r'$\kappa_{r_j\to r_i}$', transform=ax[1].transAxes, ha='right', fontsize=7, color=MUTED)
fig.tight_layout(); fig.savefig(f'{RD}/figs/fig_dynamics.pdf', bbox_inches='tight'); plt.close(fig)
def f3(x): return '%.3f' % x
best = lambda d: max((T[d][m]['mrr'], m) for m in order[:-1] if T[d][m])
qb = max((Q[n]['main']['em'], n) for n in names if Q.get(n) and n not in ('ReAct-Temporal', 'FinChrono (Qwen-7B)'))
sb = max((r[b]['auc_pr'], b) for _, b in SN if b != 'finchrono')
Nm = {'NStream': f"{R['stream']['events']:,}".replace(',', '{,}'), 'NQMain': f"{sum(q['track'] == 'main' for q in json.load(open(f'{ROOT}/qa/questions.json'))):,}".replace(',', '{,}'), 'FinDKGMRR': f3(T['findkg']['FinChrono']['mrr']), 'FinDKGBase': f3(best('findkg')[0]), 'GdeltMRR': f3(T['gdeltfin']['FinChrono']['mrr']), 'GdeltBase': f3(best('gdeltfin')[0]),
      'QAEM': f3(Q['FinChrono (Qwen-7B)']['main']['em']), 'QABase': f3(Q['ReAct-Temporal']['main']['em']), 'QARag': f3(qb[0]), 'ShockAP': f3(r['finchrono']['auc_pr']), 'ShockBase': f3(sb[0])}
A = R['abl']; ST = R['store']; hlw = R['half_life_weeks']; rb = R['robust']; bk = lambda c, k: rb[c][k]
imin = min(hlw, key=hlw.get); imax = max(hlw, key=hlw.get)
Nm.update(FDKGUnseen='%d\\%%' % round(100 * meta['findkg']['test_unseen_triple_frac']), ShockRate='%.1f\\%%' % (100 * r['finchrono']['base_rate']), FDHten=f3(T['findkg']['FinChrono']['h10']), FDTLten=f3(T['findkg']['TLogic']['h10']), FDNeural=f3(max(T['findkg'][m]['mrr'] for m in ('RE-GCN', 'TiRGN', 'HisMatch') if T['findkg'][m])), FDTLogic=f3(T['findkg']['TLogic']['mrr']), FDGain=f3(T['findkg']['FinChrono']['mrr'] - best('findkg')[0]), GDGain=f3(T['gdeltfin']['FinChrono']['mrr'] - best('gdeltfin')[0]),
  LRSFHR=f3(Q['GraphRAG/LightRAG']['main']['sfhr']), LRFLR=f3(Q['GraphRAG/LightRAG']['main']['flr']), MGFLR=f3(Q['MemGPT-FIFO']['main']['flr']), COFLR=f3(Q['CoH']['main']['flr']), LREM=f3(Q['GraphRAG/LightRAG']['main']['em']), LlEM=f3(Q['FinChrono (Llama-8B)']['main']['em']), QuEM=f3(Q['FinChrono (Qwen-14B)']['main']['em']), RSFHR=f3(Q['ReAct-Temporal']['main']['sfhr']), FSFHR=f3(Q['FinChrono (Qwen-7B)']['main']['sfhr']), FFLR=f3(Q['FinChrono (Qwen-7B)']['main']['flr']), FEv=f3(A['m7_s0']['ev']),
  AblSM=f3(A['a7_s0']['em']), AblSMsf=f3(A['a7_s0']['sfhr']), AblM=f3(A['nm7_s0']['em']), AblMev=f3(A['nm7_s0']['ev']), AblMsf=f3(A['nm7_s0']['sfhr']), AblMfl=f3(A['nm7_s0']['flr']), AblDf=f3(A['nodecay7']['f']), AblDg=f3(A['nodecay7']['g']), AblXf=f3(A['nomutual7']['f']), AblXg=f3(A['nomutual7']['g']), AblRf=f3(A['norule']['f']), AblRg=f3(A['norule']['g']), AblDs=f3(A['nodecay7']['s']), AblXs=f3(A['nomutual7']['s']),
  PaEM=f3(bk('Rewritten questions', 'para_m7')['em']), LaEM=f3(bk('30\\% filed up to 30 days late', 'late_m7')['em']), DrEM=f3(bk('20\\% of events lost', 'drop0.2_m7')['em']), DrZs=f3(bk('20\\% of events lost', 'drop0.2_zs')['em']), DrLr=f3(bk('20\\% of events lost', 'lightrag_drop0.2')['em']), DrSf=f3(bk('20\\% of events lost', 'drop0.2_m7')['sfhr']),
  StoreOver='%d\\%%' % round(100 * (1 - ST['bitemporal_ingest_ev_s'] / ST['append_ingest_ev_s'])), StoreMem='%.2f' % ST['memory_asof_ms'], StoreDuck='%.1f' % ST['duckdb_asof_ms'], HLmin='%.1f' % hlw[imin], HLmax='%.1f' % hlw[imax], HLminR=imin.replace('_', '\\_'), HLmaxR=imax.replace('_', '\\_'))
write('store', r'\begin{tabular}{lr}\toprule Measure & Value\\\midrule' + '\n' + 'Ingest, bi-temporal (events/s) & %s\\\\\n' % f"{ST['bitemporal_ingest_ev_s']:,.0f}".replace(',', '{,}') + 'Ingest, append-only (events/s) & %s\\\\\n' % f"{ST['append_ingest_ev_s']:,.0f}".replace(',', '{,}') + 'As-of read, index (ms) & %.2f\\\\\n' % ST['memory_asof_ms'] + 'As-of read, DuckDB (ms) & %.1f\\\\\n' % ST['duckdb_asof_ms'] + 'Records per read & %.1f\\\\\n' % ST['rows_per_read_mean'] + r'\bottomrule\end{tabular}' + '\n')
_qp0 = json.load(open(f'{ROOT}/qa/questions_para.json')) if os.path.exists(f'{ROOT}/qa/questions_para.json') else []
_qpt = [q for q in _qp0 if q['split'] == 'test']; Nm['RwPct'] = '%d\\%%' % round(100 * sum(q['question'] != q.get('question_orig') for q in _qpt) / max(1, len(_qpt)))
L = {d: T[d]['LogCL'] for d in T}
Nm.update(ShockGate=f3(shock('g2_nogate')['finchrono']['auc_pr']), ShockDeg=f3(r['degree']['auc_pr']), ShockGBRfifty=f3(r['gbdt']['R@50']), ShockFCRfifty=f3(r['finchrono']['R@50']), ShockHLa='%.0f' % r['target_half_lives'][0], ShockHLb='%.0f' % r['target_half_lives'][1],
  BudA=f3(R['shock_budget']['FinChrono_B2']), BudB=f3(R['shock_budget']['UniformBFS_B2']), LCfd=f3(L['findkg']['mrr']), LCgd=f3(L['gdeltfin']['mrr']), LCoffFd=f3(L['findkg']['official']['mrr']), LCoffGd=f3(L['gdeltfin']['official']['mrr']), LCnFd=f"{L['findkg']['n_test']:,}".replace(',', '{,}'), LCnGd=f"{L['gdeltfin']['n_test']:,}".replace(',', '{,}'), LCvalOff=f3(L['gdeltfin']['val_official_sample']), LCvalOne=f3(L['gdeltfin']['val']), LCsampOffGd=f3(L['gdeltfin']['official_sample']), ShockDegNoGate=f3(shock('full')['degree']['auc_pr']))
open(f'{RD}/tables/numbers.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in Nm.items()))
print(Nm)
# supporting numbers quoted in the text
R['val_curve'] = {k: summarize([x for x in (rows(k, 'val') or []) if x['q']['track'] == 'main']).get('em') for k in ('m7s0p2000', 'm7s0p4000', 'm7_s0')}
R['findkg_three_runs_without_rules'] = {f's{i}': hawkes_seed for i, hawkes_seed in enumerate([json.load(open(f'{ROOT}/hawkes/findkg/final_s{j}/result.json'))['test']['mrr'] for j in range(3)])}
R['run_args'] = {f'{d}/{t}': {k: v for k, v in json.load(open(f'{ROOT}/hawkes/{d}/{t}_s0/result.json'))['args'].items() if k in ('rules', 'opt', 'd', 'W', 'seed', 'rho_pp', 'rho_nce', 'abl')} for d in ('findkg', 'gdeltfin') for t in ('rule', 'final', 'nomutual', 'rule_nodecay')}
R['val_tkg'] = {d: {t: hawkes(d, t)['val'] for t in ('final', 'rule')} for d in ('findkg', 'gdeltfin')}
_p = f'{ROOT}/qa/questions_para.json'
if os.path.exists(_p):
    _qp = [q for q in json.load(open(_p)) if q['split'] == 'test']; R['rewrite_kept'] = dict(kept=sum(q['question'] != q.get('question_orig') for q in _qp), total=len(_qp))
_qall = json.load(open(f'{ROOT}/qa/questions.json')); R['question_counts'] = {k: sum(q['split'] == k and q['track'] == 'main' for q in _qall) for k in ('train', 'val', 'test')}; R['question_counts']['hist'] = sum(q['track'] == 'hist' for q in _qall)
json.dump(R, open(f'{RD}/results.json', 'w'), indent=1, default=float)
print('ok')

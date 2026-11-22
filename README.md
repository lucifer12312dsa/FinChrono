# FinChrono

Anonymous code and benchmark construction package for:

> **FinChrono: Bi-Temporal State-Transition Memory and Hawkes-Guided Agent Tuning for Streaming Financial Knowledge Graphs**

## Overview

FinChrono is a time-anchored memory and agent framework for streaming financial knowledge graphs built around three modules:

1. **Bi-Temporal State-Transition Store (`src/qa/store.py`)**: Types each relation as `Exclusive`, `Cumulative`, or `Transient`, executes typed state transitions per event (version closure, lineage links, period-over-period deltas, restatements, and point pulses), separates valid time from transaction (filing) time, and serves consistent as-of reads $\mathcal{G}_{\le t_q}$ via an in-memory index with DuckDB export.
2. **Continuous-Time Graph Hawkes Encoder (`src/tkg/hawkes.py`, `src/tkg/rule_feat.py`, `src/qa/hawkes_qa.py`)**: Combines a time-decayed graph attention semantic term, channel-indexed relation-specific intrinsic decay, cross-relation/two-hop excitation matrices, and a temporal rule excitation channel trained via a point-process NLL surrogate. A per-company multivariate disclosure Hawkes process ranks non-active records for QA.
3. **Temporal Operators & Paired Time-Flip Tuning (`src/qa/agent_env.py`, `src/qa/make_pairs.py`, `src/qa/train_flip.py`)**: Exposes `Snapshot_State`, `Trace_State_Lineage`, and `Emit_Temporal_Verdict` bound to anchor $t_q$, and tunes LoRA adapters (rank 64) on paired trajectories flanking real state transitions with an explicit flip-margin objective.

## Repository Structure

```text
.
├── src/
│   ├── paths.py                  # Configurable repo-relative paths (FINCHRONO_OUT / FINCHRONO_TMP / FINCHRONO_BASE)
│   ├── data/
│   │   ├── sec_pull.py           # SEC EDGAR submissions & XBRL companyfacts fetcher
│   │   ├── wikidata_pull.py      # Wikidata time-qualified company statements fetcher
│   │   └── gdelt_pull.py         # GDELT 1.0 daily financial/MNC event fetcher
│   ├── tkg/
│   │   ├── prep_tkg.py           # FinDKG and GDELT-Fin chronological 70/15/15 split builder
│   │   ├── rule_feat.py          # 1-hop and 2-hop temporal rule confidence cache
│   │   ├── hawkes.py             # Continuous-time Graph Hawkes encoder training & evaluation
│   │   └── shock.py              # Auxiliary multi-hop shock propagation evaluation script
│   ├── qa/
│   │   ├── store.py              # Bi-temporal state-transition store + DuckDB export
│   │   ├── build_stream.py       # Fin-ChronoQA bi-temporal event stream & memo builder
│   │   ├── build_questions.py    # 6-family + historical-track anchored benchmark generator
│   │   ├── hawkes_qa.py          # Per-company multivariate disclosure Hawkes scorer
│   │   ├── agent_env.py          # Temporal operators and tool-execution environment
│   │   ├── make_pairs.py         # Paired time-flip trajectory & counterfactual pair generator
│   │   ├── train_flip.py         # Paired time-flip LoRA tuning (SFT + flip-margin loss)
│   │   ├── eval_agent.py         # vLLM interactive agent evaluation (clean & degraded tracks)
│   │   ├── paraphrase.py         # Question rewriting script for robustness evaluation
│   │   ├── rag_index.py          # BGE-M3 dense embedding index builder
│   │   ├── rag_eval.py           # Dense-RAG, Time-Weighted, MemGPT-FIFO, LightRAG, GenTKG, CoH baselines
│   │   ├── train_gentkg.py       # GenTKG LoRA baseline trainer
│   │   ├── metrics.py            # EM, token F1, SFHR, FLR, and Temporal Evidence IoU
│   │   ├── rescore.py            # Unified rescoring from per-question output rows
│   │   ├── bench_store.py        # Ingest throughput and as-of query latency benchmark
│   │   └── check_store.py        # Store snapshot consistency check against independent gold timeline
│   └── analysis/
│       └── make_tables.py        # Aggregates evaluation logs into LaTeX tables, figures, and summary JSON
├── baselines/                    # Patch and runner scripts for official TKG extrapolation baselines
├── tests/
│   └── test_store.py             # Unit tests for bi-temporal state transitions and filing-date visibility
├── results/
│   └── results.json              # Aggregated benchmark results reported in the paper
├── requirements.txt
└── LICENSE
```

## Setup

Python 3.10+ and PyTorch (with CUDA for GPU training/inference) are required.

```bash
pip install -r requirements.txt
```

All scripts default to repo-relative directories (`./out`, `./tmp`, `./external_baselines`) defined in `src/paths.py`. You can override them with environment variables:

```bash
export FINCHRONO_OUT=/path/to/out
export FINCHRONO_TMP=/path/to/tmp
export FINCHRONO_BASE=/path/to/external_baselines
```

### Quick Verification (No External Data or GPU Needed)

Run the bi-temporal state machine unit tests covering exclusive flips, explicit closures, repeated statements, out-of-order late filings, cumulative deltas and restatements, transient filing visibility, and the append-only ablation:

```bash
python tests/test_store.py
```

## Reproducing Experiments

### 1. Data Acquisition & Benchmark Construction

```bash
# Public sources (SEC EDGAR, Wikidata, GDELT 1.0)
python src/data/sec_pull.py
python src/data/wikidata_pull.py
python src/data/gdelt_pull.py

# Temporal link extrapolation splits (FinDKG & GDELT-Fin, chronological 70/15/15)
python src/tkg/prep_tkg.py

# Fin-ChronoQA bi-temporal stream (621,951 events) & 6-family + historical benchmark
python src/qa/build_stream.py
python src/qa/build_questions.py
python src/qa/check_store.py
python src/qa/bench_store.py
```

### 2. Temporal Link Extrapolation (Table 2 & Fig. 3)

```bash
# Precompute 1-hop / 2-hop temporal rule features
python src/tkg/rule_feat.py --data findkg
python src/tkg/rule_feat.py --data gdeltfin

# Train and evaluate FinChrono Graph Hawkes Encoder (time-aware filtered protocol)
python src/tkg/hawkes.py --data findkg --tag rule --rules 1 --seed 0
python src/tkg/hawkes.py --data gdeltfin --tag rule --rules 1 --seed 0
```

See `baselines/README.md` for running the six official TKG extrapolation baselines (RE-GCN, TiRGN, HiSMatch, TLogic, KGTransformer, LogCL) under the same splits and time-aware filtered protocol.

### 3. Time-Anchored Question Answering (Tables 3–6 & Fig. 2)

```bash
# Fit per-company disclosure Hawkes scorer
python src/qa/hawkes_qa.py

# Generate paired time-flip trajectories (20,000 pairs)
python src/qa/make_pairs.py --view bitemp --n 20000

# Train FinChrono agent policy (Qwen2.5-7B-Instruct, LoRA rank 64, lambda_m=0.5, delta=2.0)
python src/qa/train_flip.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --pairs qa/pairs_bitemp_h.jsonl \
  --out sft/m7_s0 \
  --n_pairs 6000 --lam 0.5 --delta 2.0 --rank 64 --seed 0

# Evaluate FinChrono and untuned ReAct-Temporal
python src/qa/eval_agent.py --name m7_s0 --model Qwen/Qwen2.5-7B-Instruct --lora sft/m7_s0/lora --split test --tracks main,hist
python src/qa/eval_agent.py --name zs7 --model Qwen/Qwen2.5-7B-Instruct --zeroshot 1 --split test --tracks main,hist

# Evaluate memory baselines (Dense-RAG, Time-Weighted, MemGPT-FIFO, GraphRAG/LightRAG, GenTKG, CoH)
python src/qa/rag_index.py
python src/qa/train_gentkg.py
python src/qa/rag_eval.py --split test --tracks main,hist

# Robustness checks (rewritten questions, 10%/20% dropped events, 30% late filings)
python src/qa/paraphrase.py
python src/qa/eval_agent.py --name para_m7 --lora sft/m7_s0/lora --qfile questions_para.json --tracks main
python src/qa/eval_agent.py --name drop0.2_m7 --lora sft/m7_s0/lora --drop 0.2 --tracks main
python src/qa/eval_agent.py --name late_m7 --lora sft/m7_s0/lora --delay 0.3 --tracks main

# Unified rescoring and paper table/figure generation
python src/qa/rescore.py
python src/analysis/make_tables.py
```

## Summary of Reported Results

Aggregated metrics are also stored in `results/results.json`:

| Benchmark | Metric | Best Baseline | FinChrono |
| :--- | :--- | :---: | :---: |
| FinDKG (test) | MRR (filtered) | 0.222 (TLogic) | **0.230** |
| GDELT-Fin (test) | MRR (filtered) | 0.428 (TiRGN) | **0.459** |
| Fin-ChronoQA (test) | Exact Match | 0.869 (ReAct-Temporal) / 0.719 (GenTKG) | **0.991** |
| Fin-ChronoQA (test) | SFHR (lower is better) | 0.046 (ReAct-Temporal) | **0.006** |
| Fin-ChronoQA (test) | FLR (lower is better) | 0.000 (ReAct-Temporal) | **0.000** |

## License

Released under the MIT License (see `LICENSE`).

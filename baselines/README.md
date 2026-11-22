# Temporal Knowledge Graph Extrapolation Baselines

This directory contains setup patches and runner scripts for the six official TKG baselines evaluated under the identical chronological 70/15/15 splits (`findkg` and `gdeltfin`) and time-aware filtered per-query protocol:

- **RE-GCN** (`setup_tkg.sh`, `run_regcn.sh`)
- **TiRGN** (`setup_tkg.sh`, `run_tirgn.sh`)
- **HiSMatch** (`setup_tkg.sh`, `run_hismatch.sh`)
- **TLogic** (`run_tlogic.sh`, `run_tlogic_apply.sh`)
- **KGTransformer / FinDKG** (`setup_kgt.sh`, `run_kgt.sh`, `fc_compat.py`)
- **LogCL** (`setup_logcl.sh`, `run_logcl.sh`, `run_logcl_single.sh`, `patch_logcl_single.py`)

## Environment Variables

All scripts default to:
- `FINCHRONO_OUT=<repo>/out` (expects `$FINCHRONO_OUT/tkg/{findkg,gdeltfin}` created by `src/tkg/prep_tkg.py`)
- `FINCHRONO_BASE=<repo>/external_baselines` (clones and patches official baseline repositories here)
- `TKG_ENV_ACTIVATE` (optional path to a virtualenv/conda activation script for DGL-based baselines)

## Notes on Patches

- `fc_compat.py`: Moves GPU index tensors to CPU when indexing CPU tensors in official FinDKG code (required on PyTorch >= 1.9; numerical values are unchanged).
- `patch_logcl_single.py`: Implements per-query time-aware filtered evaluation (`FC_SINGLE`) for LogCL so each unique `(s, r)` query uses only its own query mask and history subgraph rather than pooling all queries within a snapshot.

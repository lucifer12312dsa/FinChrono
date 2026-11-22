"""Centralized path configuration for FinChrono.

All scripts default to repo-relative directories (`./out`, `./tmp`, `./external_baselines`)
and can be overridden via environment variables:
  - FINCHRONO_OUT  : output/data root directory (default: <repo>/out)
  - FINCHRONO_TMP  : temporary download/cache directory (default: <repo>/tmp)
  - FINCHRONO_BASE : directory for cloned third-party TKG baselines (default: <repo>/external_baselines)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("FINCHRONO_OUT", str(REPO_ROOT / "out"))).resolve()
TMP_DIR = Path(os.environ.get("FINCHRONO_TMP", str(REPO_ROOT / "tmp"))).resolve()
BASE_DIR = Path(os.environ.get("FINCHRONO_BASE", str(REPO_ROOT / "external_baselines"))).resolve()

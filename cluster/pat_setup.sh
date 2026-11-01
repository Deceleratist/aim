#!/bin/bash
# One-time setup of AIM on Pat (run ON Pat, on the login node):  bash pat_setup.sh
# - code:   ~/aim                         (clone of github.com/Deceleratist/aim, branch jz/benchmark-wsl)
# - env:    ~/aim-env                     (Python venv, same pinned versions as the laptop WSL env)
# - runs:   /sharedscratch/jz636/aim_runs (job outputs; big, so not in home)
# Safe to re-run: it skips what already exists.
set -euo pipefail

BRANCH=jz/benchmark-wsl
RUNS=/sharedscratch/$USER/aim_runs
mkdir -p "$RUNS"

# 1. code
if [ ! -d ~/aim ]; then
    git clone -q --branch "$BRANCH" https://github.com/Deceleratist/aim.git ~/aim
    git -C ~/aim remote rename origin fork
    git -C ~/aim remote add origin https://github.com/RD-alpha/aim.git
    git -C ~/aim remote set-url --push origin DISABLED-push-to-fork-instead
fi
git -C ~/aim fetch -q fork && git -C ~/aim status -sb | head -1

# 2. a Python >= 3.10 (RDKit 2025.3 wheels need it). Prefer the system one, then modules, then uv.
#    Pat (Nov 2026): system python is 3.9 and the anaconda module is a no-op, so pass a base
#    interpreter explicitly, e.g. BASE_PYTHON=~/venvs/n6q311/bin/python3.11 bash pat_setup.sh
#    (only its interpreter is used; that env itself is not modified).
PY="${BASE_PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
            PY=$(command -v "$cand"); break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "No Python >= 3.10 found; set BASE_PYTHON=/path/to/python3.1x and re-run"; exit 1
fi
echo "using $PY ($($PY --version))"

# 3. env
if [ ! -d ~/aim-env ]; then "$PY" -m venv ~/aim-env; fi
source ~/aim-env/bin/activate
pip install -q --upgrade pip
pip install -q rdkit==2025.3.2 xgboost==3.0.2 scikit-learn==1.7.0 anytree==2.13.0 numpy pandas matplotlib pytest

# 4. check
cd ~/aim && python -m pytest -q tests | tail -1
python -c "import rdkit, xgboost, sklearn; print('aim-env OK', rdkit.__version__, xgboost.__version__, sklearn.__version__)"
echo "runs folder: $RUNS"

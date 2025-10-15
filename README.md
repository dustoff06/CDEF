# CDEF Analyzer — Concordance–Dispersion–Extremeness for Ranking Data

Copula-powered ranking diagnostics that **jointly** model:

- **Concordance** (Kendall’s W)
- **Concurrence** (mutual information / dependence)
- **Extremeness** (upper-tail behavior via a Gumbel copula)

with automatic detection of **forced vs non-forced** rankings and data-driven choice of the dispersion regime (Mallows for forced, dependent data; multinomial or uniform baselines otherwise).

> Paper code and figure assets for: **The Concordance–Dispersion–Extremeness Framework (CDEF)**  
> Repo: <https://github.com/dustoff06/CDEF>

---

## Key features

- Upper-tail dependence via **Gumbel** copulas on **mid-rank** pseudo-observations `(r - 0.5) / N`.
- **Auto-detect** forced vs non-forced rankings.
- **Mallows** scoring for forced rankings under dependence.
- Likelihood summaries and an **interpretation rule** that flags _genuine_ vs _phantom_ concordance.
- Reproducible **exemplar simulations**: Phantom, Genuine, Random, Clustered.
- Figure assets and generation code included.

```bash
CDEF/
├─ docs/
│  └─ figures/
│     └─ Figure 1.png
├─ src/
│  ├─ cdef_analyzer/
│  │  ├─ __init__.py
│  │  └─ core.py            # library API (RankDependencyAnalyzer, helpers)
│  ├─ examples/
│  │  ├─ data.xlsx          # exemplar input (Rater, Ratee, Ranking)
│  │  ├─ demo.py            # programmatic exemplar scenarios
│  │  └─ run_demo.py        # CLI-style entry script
│  └─ tests/
│     └─ test_imports.py
├─ README.md
├─ requirements.txt
└─ pyproject.toml

```

## Installation

Tested on **Python 3.12** (Linux/macOS/Windows; examples show WSL paths but any OS path works).

```bash
# Clone and create a virtual environment
git clone https://github.com/dustoff06/CDEF.git
cd CDEF

python -m venv .venv            # cross-platform; avoids hardcoding version
# Windows PowerShell:
#   .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Upgrade build tooling
pip install -U pip setuptools wheel

# Install dependencies and the package (editable dev install)
pip install -r requirements.txt
pip install -e .               

# Quick check (imports public API re-exports)
python - <<'PY'
from cdef_analyzer import RankDependencyAnalyzer, CopulaResults
print("OK:", RankDependencyAnalyzer, CopulaResults)
PY

```

# Quick Start (Option A)

```bash

python -m src.examples.run_demo \
  --excel src/examples/data.xlsx \
  --sheet Sheet1 \
  --rater Rater --ratee Ratee --ranking Ranking \
  --seed 42
```

You’ll see:

  ranking type (forced / non-forced)
  
  selected distribution model (e.g., Mallows)
  
  Kendall’s W, theta (scaled), theta (from tau), mean Kendall’s tau, mutual information
  
  copula average log-likelihood, independence baseline
  
  relative-importance weights
  
  pairwise theta range
  
  a CDEF interpretation label (e.g., “GENUINE: Natural agreement”)

# Option B-Python API

```bash
from cdef_analyzer.gumbel_copula_fixed import RankDependencyAnalyzer, format_results

an = RankDependencyAnalyzer(random_seed=42)
results = an.analyze_from_excel(
    file_path="src/examples/data.xlsx",
    sheet_name="Sheet1",
    rater_col="Rater",
    ratee_col="Ratee",
    ranking_col="Ranking",
)
print(format_results(results))

```

# Option C-Reproduce Exemplars

```bash

python src/examples/run_demo.py
```

This generates the Phantom / Genuine / Random / Clustered scenarios, writes scenario Excel files, prints summaries, and saves a comparison table cdef_summary_fixed.csv. The interpretation rule maps (W, theta, MI, mean_tau) to a narrative class and P(Genuine | Data).

# Data format

Input is long format with columns:
    
    Rater — source of the ranking
    
    Ratee — item/team/entity being ranked
    
    Ranking — integer rank

The analyzer internally pivots to wide form and auto-detects forced vs non-forced rankings.

# Determinism and environment

Python: 3.9

Random seeds: numpy seed set to 42 for exemplars; the analyzer accepts a random_seed at initialization.

Paths: Examples show WSL-style mounts, but standard Windows/macOS/Linux paths work.

# API Sketch
```bash
from cdef_analyzer import *
an = RankDependencyAnalyzer(
    num_samples=10000,        # for Monte Carlo/permutation tasks if used
    significance_level=0.05,  # chi-square threshold
    random_seed=42
)

results = an.analyze_from_excel(
    file_path="src/examples/data.xlsx",
    sheet_name="Sheet1",
    rater_col="Rater",
    ratee_col="Ratee",
    ranking_col="Ranking",
)
results
```
results contains (non-exhaustive):

theta_scaled, theta_gumbel, kendalls_W, avg_kendalls_tau

mutual_information, chi_square_stat, p_value

avg_log_likelihood, independence_log_likelihood

pairwise_thetas (dict)

tau_matrix (pandas DataFrame)

ranking_type, distribution_model, model_log_likelihood

relative_importance (dict)

n_raters, n_items

# Helper
```bash
from cdef_analyzer.core import format_results
print(format_results(results))
```

# FAQ

FAQ

Do I need a GPU?
No. Everything runs on CPU.

Can the analyzer handle ties?
Yes. It detects non-forced rankings and adapts the modeling path.

I have large data, anything to tune?
Reduce verbose printing, skip writing per-scenario Excel files, and consider batching pairwise copula fits.



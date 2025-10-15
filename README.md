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
│  └─ Figure_1.png
├─ examples/
│  ├─ data.xlsx              # exemplar input (Rater, Ratee, Ranking)
│  └─ run_demo.py            # 4-scenario exemplar script
├─ src/
│  └─ cdef_analyzer/
│     ├─ __init__.py         # re-exports: RankDependencyAnalyzer, CopulaResults
│     ├─ _version.py
│     ├─ demo.py             # CLI entry module (format_summary, write_csv, main)
│     └─ gumbel_copula_fixed.py  # library implementation
├─ tests/
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

python -m venv .venv            # cross-platform; no hardcoded version
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Upgrade build tooling
python -m pip install -U pip setuptools wheel

# Install dependencies
python -m pip install -r requirements.txt
```

# Quick Start (Option A)

This example runs the NCAA analysis from the paper.

```bash

python -m cdef_analyzer.demo \
  --excel examples/data.xlsx \
  --sheet Sheet1 \
  --rater-col Rater --ratee-col Ratee --ranking-col Ranking \
  --seed 42
```

You’ll see:

- ranking type (forced / non-forced)
- selected distribution model (e.g., Mallows)
- Kendall’s W, θ (scaled), θ (from τ), mean Kendall’s τ, mutual information
- copula average log-likelihood, independence baseline
- relative-importance weights
- pairwise θ range
- CDEF interpretation label (e.g., “GENUINE: Natural agreement”)


# Option B-Python API

This example runs the NCAA analysis from the paper.

```bash
from cdef_analyzer import RankDependencyAnalyzer, CopulaResults
from cdef_analyzer.demo import format_summary  # pretty printer

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

This example runs the exemplars from the paper.

```bash

python examples/run_demo.py 

```

This generates the Phantom / Genuine / Random / Clustered scenarios, writes scenario Excel files, prints summaries, and saves a comparison table cdef_summary_fixed.csv. The interpretation rule maps (W, theta, MI, mean_tau) to a narrative class and P(Genuine | Data).

# Data format

Input is long format with columns:
    
    Rater — source of the ranking
    
    Ratee — item/team/entity being ranked
    
    Ranking — integer rank

The analyzer internally pivots to wide form and auto-detects forced vs non-forced rankings.

# Determinism and environment

Python: 3.12

Random seeds: numpy seed set to 42 for exemplars; the analyzer accepts a random_seed at initialization.

Paths: Examples show WSL-style mounts, but standard Windows/macOS/Linux paths work.

# API Sketch
```bash
from cdef_analyzer import RankDependencyAnalyzer, CopulaResults
from cdef_analyzer.demo import format_summary, write_csv  # helpers for pretty print / CSV

an = RankDependencyAnalyzer(random_seed=42)

res: CopulaResults = an.analyze_from_excel(
    file_path="examples/data.xlsx",   # moved out of src/
    sheet_name="Sheet1",
    rater_col="Rater",
    ratee_col="Ratee",
    ranking_col="Ranking",
)

print(format_summary(res))            # nice human-readable report
# write_csv(res, "ncaa_summary.csv")  # optional: save headline metrics

```
Results contains (non-exhaustive):

- `theta_scaled`, `theta_gumbel`, `kendalls_W`, `avg_kendalls_tau`
- `mutual_information`, `chi_square_stat`, `p_value`
- `avg_log_likelihood`, `independence_log_likelihood`
- `pairwise_thetas` *(dict)*
- `tau_matrix` *(pandas.DataFrame)*
- `ranking_type`, `distribution_model`, `model_log_likelihood`
- `relative_importance` *(dict)*
- `n_raters`, `n_items`


# FAQ

FAQ

Do I need a GPU?
No. Everything runs on CPU.

Can the analyzer handle ties?
Yes. It detects non-forced rankings and adapts the modeling path.

I have large data, anything to tune?
Reduce verbose printing, skip writing per-scenario Excel files, and consider batching pairwise copula fits.



# core.py
"""
CDEF core library: one import surface for analysis, scenarios, and I/O.

- analyze_excel(...) -> dict      : run analyzer on long-format Excel
- analyze_dataframe(...) -> dict  : run analyzer on a pandas DataFrame
- run_demo_scenarios(...) -> pd.DataFrame : reproduce the 4-scenario demo
- results_to_dict(...) -> dict    : serialize CopulaResults for JSON
- write_requirements(...)         : emit requirements.txt pinned for Py3.9

Relies on your existing implementation files:
- gumbel_copula_fixed.py (analyzer, results dataclass)
- cdef_demo_fixed.py     (scenario generators & demo runner)
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

import os
import pandas as pd

# --- Import your analyzer & demo code ---------------------------------------
# (These come from the two files you shared.)
from gumbel_copula_fixed import RankDependencyAnalyzer, CopulaResults  # noqa: E402
# ^ Source: gumbel_copula_fixed.py  :contentReference[oaicite:2]{index=2}

# The demo module defines scenario builders and a runner;
# we import lazily so core works even if users only want the analyzer.
try:
    import cdef_demo_fixed as demo  # noqa: E402
    # ^ Source: cdef_demo_fixed.py  :contentReference[oaicite:3]{index=3}
except Exception:
    demo = None


# --- Public helpers ----------------------------------------------------------

def results_to_dict(results: CopulaResults) -> Dict[str, Any]:
    """
    Convert CopulaResults (dataclass) to a plain dict with JSON-friendly fields.
    """
    d = asdict(results)
    # Ensure DataFrame is JSON-friendly (stringify or records)
    d["tau_matrix"] = results.tau_matrix.to_dict(orient="split")
    return d


def analyze_excel(
    file_path: str,
    sheet_name: str = "Sheet1",
    rater_col: str = "Rater",
    ratee_col: str = "Ratee",
    ranking_col: str = "Ranking",
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Load long-format Excel, pivot to wide, and run the analyzer pipeline.
    Returns a JSON-serializable dict.
    """
    analyzer = RankDependencyAnalyzer(random_seed=random_seed)
    results = analyzer.analyze_from_excel(
        file_path=file_path,
        sheet_name=sheet_name,
        rater_col=rater_col,
        ratee_col=ratee_col,
        ranking_col=ranking_col,
    )
    return results_to_dict(results)


def analyze_dataframe(
    df_long: pd.DataFrame,
    rater_col: str = "Rater",
    ratee_col: str = "Ratee",
    ranking_col: str = "Ranking",
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Run the analyzer when you already have a long-format DataFrame in memory.
    Mirrors the Excel path used in gumbel_copula_fixed but avoids disk I/O.
    """
    # Reuse analyzer's load/pivot logic by writing a tiny temp file, or pivot here:
    # Pivot long -> wide (ratees as rows, raters as columns)
    rankings_wide = df_long.pivot(index=ratee_col, columns=rater_col, values=ranking_col)
    rankings_wide = rankings_wide.dropna().astype(int)

    analyzer = RankDependencyAnalyzer(random_seed=random_seed)

    # --- Inline of analyzer steps to match analyze_from_excel() semantics ---
    analyzer.fit_gumbel_copulas(rankings_wide)
    theta_scaled = analyzer.estimate_copula_theta(rankings_wide)
    theta_gumbel = analyzer.estimate_gumbel_theta(rankings_wide)

    # Mutual information & chi-square on first two raters (consistent with impl)
    col1, col2 = rankings_wide.columns[0], rankings_wide.columns[1]
    mi, p_value, chi2_stat = analyzer.compute_mutual_information_and_independence(
        rankings_wide[col1].values, rankings_wide[col2].values
    )

    distribution_model, model_log_likelihood = analyzer.choose_distribution_model(rankings_wide)
    avg_log_likelihood = analyzer.compute_avg_log_likelihood(rankings_wide)
    independence_ll = analyzer.compute_independence_log_likelihood(rankings_wide)
    relative_importance = analyzer.compute_relative_importance(analyzer.W, mi, theta_scaled)

    # Pairwise tau & thetas
    from scipy.stats import kendalltau  # local import to keep deps obvious
    taus = []
    for i, c1 in enumerate(rankings_wide.columns):
        for c2 in rankings_wide.columns[i + 1:]:
            tau, _ = kendalltau(rankings_wide[c1], rankings_wide[c2])
            taus.append(tau)
    avg_tau = round(float(sum(taus) / len(taus)), 3) if taus else 0.0

    pairwise_thetas = {pair: round(c.theta, 3) for pair, c in analyzer.copulas.items()}

    tau_matrix = pd.DataFrame(index=rankings_wide.columns, columns=rankings_wide.columns, dtype=float)
    for c1 in rankings_wide.columns:
        for c2 in rankings_wide.columns:
            if c1 == c2:
                tau_matrix.loc[c1, c2] = 1.0
            else:
                tau, _ = kendalltau(rankings_wide[c1], rankings_wide[c2])
                tau_matrix.loc[c1, c2] = round(tau, 3)

    results = CopulaResults(
        theta_scaled=theta_scaled,
        theta_gumbel=theta_gumbel,
        kendalls_W=analyzer.W,
        avg_kendalls_tau=avg_tau,
        mutual_information=mi,
        chi_square_stat=chi2_stat,
        p_value=p_value,
        avg_log_likelihood=avg_log_likelihood,
        independence_log_likelihood=independence_ll,
        pairwise_thetas=pairwise_thetas,
        tau_matrix=tau_matrix,
        ranking_type=analyzer.detect_ranking_type(rankings_wide),
        distribution_model=distribution_model,
        model_log_likelihood=model_log_likelihood,
        relative_importance=relative_importance,
        n_raters=len(rankings_wide.columns),
        n_items=len(rankings_wide),
    )
    return results_to_dict(results)


def run_demo_scenarios(
    outdir: Optional[str] = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """
    Reproduce the 4-scenario demonstration and return the summary DataFrame.
    If 'outdir' is provided, also write cdef_summary_fixed.csv there.
    """
    if demo is None:
        raise RuntimeError(
            "cdef_demo_fixed.py not found; cannot run scenarios. "
            "Make sure the module is available alongside core.py."
        )
    # Borrow the demo's runner to keep behavior identical to the paper’s appendix.
    df_results = demo.run_cdef_demonstration()  # prints console report as in your script
    csv_path = ""
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, "cdef_summary_fixed.csv")
        df_results.to_csv(csv_path, index=False)
    return df_results, csv_path


def write_requirements(path: str = "requirements.txt", include_api: bool = True) -> str:
    """
    Emit a conservative requirements.txt pinned for Python 3.9 runtime.
    include_api=True also adds FastAPI/uvicorn for the optional web API.
    """
    base = [
        # Core numerical / data stack (Py3.9-compatible pins)
        "numpy>=1.23,<2.0",
        "pandas>=1.5,<2.1",
        "scipy>=1.9,<1.12",
        # Copulas library used by the analyzer
        "copulas>=0.8.0,<0.9",
        # Excel I/O used in the demo & analyzer loader
        "openpyxl>=3.0.10,<4.0",
    ]
    api = [
        "fastapi>=0.95,<1.0",
        "uvicorn[standard]>=0.22,<1.0",
        "pydantic>=1.10,<2.0",
        # Optional file upload parsing in many tutorials; harmless to include
        "python-multipart>=0.0.6,<0.0.8",
    ]
    lines = base + (api if include_api else [])
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Python 3.9 environment\n")
        f.write("\n".join(lines) + "\n")
    return os.path.abspath(path)


# --- Optional: tiny CLI for quick smoke-tests --------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="CDEF core CLI")
    p.add_argument("--excel", type=str, help="Path to long-format Excel")
    p.add_argument("--sheet", type=str, default="Sheet1")
    p.add_argument("--rater", type=str, default="Rater")
    p.add_argument("--ratee", type=str, default="Ratee")
    p.add_argument("--ranking", type=str, default="Ranking")
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--write-reqs", action="store_true")
    args = p.parse_args()

    if args.write_reqs:
        path = write_requirements(include_api=True)
        print(f"requirements.txt written to {path}")

    if args.excel:
        result = analyze_excel(
            file_path=args.excel,
            sheet_name=args.sheet,
            rater_col=args.rater,
            ratee_col=args.ratee,
            ranking_col=args.ranking,
        )
        import json
        print(json.dumps(result, indent=2))
    elif args.outdir:
        df, csv_path = run_demo_scenarios(outdir=args.outdir)
        print(f"Demo summary saved to: {csv_path if csv_path else '(not written)'}")
    else:
        p.print_help()

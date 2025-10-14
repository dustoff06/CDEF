# core.py
"""
CDEF core library: one import surface for analysis, scenarios, and I/O.
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

import os
import pandas as pd
from scipy.stats import kendalltau

# The demo module (optional)
try:
    from . import cdef_demo_fixed as demo
except Exception:
    demo = None


def results_to_dict(results: CopulaResults) -> Dict[str, Any]:
    """Convert CopulaResults to JSON-friendly dict."""
    d = asdict(results)
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
    """Load Excel and run analyzer."""
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
    """Run analyzer on DataFrame."""
    rankings_wide = df_long.pivot(index=ratee_col, columns=rater_col, values=ranking_col)
    rankings_wide = rankings_wide.dropna().astype(int)

    analyzer = RankDependencyAnalyzer(random_seed=random_seed)
    analyzer.fit_gumbel_copulas(rankings_wide)
    
    theta_scaled = analyzer.estimate_copula_theta(rankings_wide)
    theta_gumbel = analyzer.estimate_gumbel_theta(rankings_wide)

    col1, col2 = rankings_wide.columns[0], rankings_wide.columns[1]
    mi, p_value, chi2_stat = analyzer.compute_mutual_information_and_independence(
        rankings_wide[col1].values, rankings_wide[col2].values
    )

    distribution_model, model_log_likelihood = analyzer.choose_distribution_model(rankings_wide)
    avg_log_likelihood = analyzer.compute_avg_log_likelihood(rankings_wide)
    independence_ll = analyzer.compute_independence_log_likelihood(rankings_wide)
    relative_importance = analyzer.compute_relative_importance(analyzer.W, mi, theta_scaled)

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
    """Run the 4-scenario demonstration."""
    if demo is None:
        raise RuntimeError("cdef_demo_fixed.py not found")
    
    df_results = demo.run_cdef_demonstration()
    csv_path = ""
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, "cdef_summary_fixed.csv")
        df_results.to_csv(csv_path, index=False)
    return df_results, csv_path

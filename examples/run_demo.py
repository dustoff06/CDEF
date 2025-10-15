#!/usr/bin/env python3
"""
CDEF Demonstration: Detecting Phantom vs Genuine Concordance

Reproduces four synthetic scenarios using the packaged analyzer:
- Phantom (Extreme Bias)
- Genuine (Natural Agreement)
- Random (No Agreement)
- Clustered (Outlier rater)

Design choices:
- Converts wide rater×team rankings to the analyzer's expected long Excel format.
- Calls RankDependencyAnalyzer.analyze_from_excel (same path as CLI) for parity.
- Interprets results using W (concordance), θ (tail dependence), MI (concurrence), and τ (avg Kendall's tau).
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .src.cdef_analyzer import RankDependencyAnalyzer, CopulaResults


# ---------------------------
# Scenario generators (N teams × R raters)
# ---------------------------

def _alt_extreme_pattern(n_teams: int) -> list[int]:
    """Alternate top and bottom ranks: 1, n, 2, n-1, ..."""
    pattern = []
    for i in range(n_teams // 2):
        pattern.append(i + 1)          # top
        pattern.append(n_teams - i)    # bottom
    if n_teams % 2 == 1:
        pattern.append(n_teams // 2 + 1)
    return pattern


def create_phantom_scenario(n_teams: int = 136, rater_names=('CBS', 'CFN', 'Congrove', 'NYT')) -> pd.DataFrame:
    """
    HIGH W + VERY HIGH theta via shared extreme-bias pattern with tiny perturbations.
    """
    base = _alt_extreme_pattern(n_teams)
    rankings: Dict[str, list[int]] = {}
    for rater in rater_names:
        arr = base.copy()
        for _ in range(2):  # small perturbations to avoid perfect ties
            i, j = np.random.choice(n_teams, 2, replace=False)
            arr[i], arr[j] = arr[j], arr[i]
        rankings[rater] = arr
    df = pd.DataFrame(rankings)
    df.index.name = "Team"
    return df


def create_genuine_scenario(n_teams: int = 136, rater_names=('CBS', 'CFN', 'Congrove', 'NYT')) -> pd.DataFrame:
    """
    HIGH W + MODERATE theta via natural agreement plus moderate noise.
    """
    base = list(range(1, n_teams + 1))
    rankings: Dict[str, list[int]] = {}
    for rater in rater_names:
        arr = base.copy()
        for _ in range(12):  # moderate swaps
            i, j = np.random.choice(n_teams, 2, replace=False)
            arr[i], arr[j] = arr[j], arr[i]
        rankings[rater] = arr
    df = pd.DataFrame(rankings)
    df.index.name = "Team"
    return df


def create_random_scenario(n_teams: int = 136, rater_names=('CBS', 'CFN', 'Congrove', 'NYT')) -> pd.DataFrame:
    """
    LOW W + LOW theta via independence (random permutations).
    """
    rankings: Dict[str, list[int]] = {}
    for rater in rater_names:
        rankings[rater] = list(np.random.permutation(range(1, n_teams + 1)))
    df = pd.DataFrame(rankings)
    df.index.name = "Team"
    return df


def create_clustered_scenario(n_teams: int = 136) -> pd.DataFrame:
    """
    HIGH W among 3 raters + one divergent rater (mimics a "Congrove" outlier).
    """
    base = list(range(1, n_teams + 1))
    rankings: Dict[str, list[int]] = {}

    # A tight cluster of three raters
    for rater in ('CBS', 'CFN', 'NYT'):
        arr = base.copy()
        for _ in range(8):  # small noise
            i, j = np.random.choice(n_teams, 2, replace=False)
            arr[i], arr[j] = arr[j], arr[i]
        rankings[rater] = arr

    # One divergent rater with much larger noise
    arr = base.copy()
    for _ in range(40):
        i, j = np.random.choice(n_teams, 2, replace=False)
        arr[i], arr[j] = arr[j], arr[i]
    rankings['Congrove'] = arr

    df = pd.DataFrame(rankings)
    df.index.name = "Team"
    return df


# ---------------------------
# Interpretation layer
# ---------------------------

def cdef_interpretation(results: CopulaResults) -> Tuple[float, str]:
    """
    Classify using W (concordance), θ (tail dependence), MI (concurrence), and τ (avg Kendall's tau).
    Returns (probability_genuine, interpretation_string).
    """
    W = results.kendalls_W
    theta = results.theta_scaled
    mi = results.mutual_information
    avg_tau = results.avg_kendalls_tau

    if W > 0.85 and theta > 15.0:
        return 0.10, "⚠️ PHANTOM: Shared extreme biases (very high W + very high θ)"
    elif W > 0.75 and theta > 8.0:
        return 0.20, "⚠️ Likely PHANTOM: High concordance with extreme tail dependence"
    elif W > 0.75 and theta < 4.0:
        return 0.85, "✓✓ STRONG GENUINE: Excellent natural agreement"
    elif W > 0.65 and 2.5 < theta < 6.5:
        return 0.75, "✓ GENUINE: Natural agreement (high W + moderate θ)"
    elif W > 0.5 and 0.3 < avg_tau < 0.65:
        return 0.60, "→ CLUSTERED: Subgroup agreement with divergent rater(s)"
    elif W > 0.25:
        return 0.70, "◐ WEAK: Limited systematic agreement"
    else:
        return 0.90, "○ RANDOM: No systematic agreement (independence)"


# ---------------------------
# Utilities
# ---------------------------

def to_long_format(rank_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide rater×team rankings to the long format expected by the analyzer."""
    rows = []
    n = len(rank_df)
    for team_idx in range(n):
        for rater in rank_df.columns:
            rows.append({
                "Ratee": f"Team_{team_idx + 1}",
                "Rater": rater,
                "Ranking": int(rank_df.iloc[team_idx][rater]),
            })
    return pd.DataFrame(rows)


def analyze_rank_df(analyzer: RankDependencyAnalyzer, rank_df: pd.DataFrame) -> CopulaResults:
    """
    Persist to a temp Excel file (Sheet1) and call analyze_from_excel to match CLI exactly.
    """
    long_df = to_long_format(rank_df)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scenario.xlsx")
        long_df.to_excel(path, sheet_name="Sheet1", index=False)
        return analyzer.analyze_from_excel(
            file_path=path,
            sheet_name="Sheet1",
            rater_col="Rater",
            ratee_col="Ratee",
            ranking_col="Ranking",
        )


# ---------------------------
# Main demonstration
# ---------------------------

def main() -> int:
    np.random.seed(42)

    print("\n" + "=" * 80)
    print("CDEF DEMONSTRATION: Detecting Phantom vs Genuine Concordance")
    print("Using Properly Fixed Gumbel Copula Analysis")
    print("=" * 80)

    scenarios = {
        "Phantom (Extreme Bias)": create_phantom_scenario(),
        "Genuine (Natural Agreement)": create_genuine_scenario(),
        "Random (No Agreement)": create_random_scenario(),
        "Clustered (Outlier)": create_clustered_scenario(),
    }

    analyzer = RankDependencyAnalyzer(random_seed=42)
    collected = []

    for name, rank_df in scenarios.items():
        print(f"\n{'=' * 80}\n{name.upper()}\n{'=' * 80}")
        res = analyze_rank_df(analyzer, rank_df)

        # Traditional summary (W only)
        if res.kendalls_W > 0.7:
            traditional = "High concordance → Good agreement"
        elif res.kendalls_W > 0.4:
            traditional = "Moderate concordance → Some agreement"
        else:
            traditional = "Low concordance → Poor agreement"

        p_gen, interp = cdef_interpretation(res)

        print(f"\nRanking Type: {res.ranking_type}")
        print(f"Distribution Model: {res.distribution_model}")
        if res.model_log_likelihood is not None:
            print(f"Model Log-Likelihood: {res.model_log_likelihood}")

        print("\nCore Metrics:")
        print(f"  Kendall's W (concord.)  : {res.kendalls_W:.3f}")
        print(f"  θ (scaled)               : {res.theta_scaled:.3f}")
        print(f"  θ (from τ)               : {res.theta_gumbel:.3f}")
        print(f"  Avg Kendall's τ          : {res.avg_kendalls_tau:.3f}")
        print(f"  Mutual information       : {res.mutual_information:.3f}")

        print("\nLog-Likelihoods:")
        print(f"  Copula (average)         : {res.avg_log_likelihood:.6f}")
        print(f"  Independence baseline    : {res.independence_log_likelihood:.6f}")

        print("\nRelative Importance (weights, not probabilities)")
        for k, v in res.relative_importance.items():
            print(f"  {k:15s}: {v:.3f}")

        if res.pairwise_thetas:
            thetas = list(res.pairwise_thetas.values())
            print("\nPairwise θ (Gumbel) — range")
            print(f"  min={min(thetas):.3f}, max={max(thetas):.3f}")

        print("\nTraditional (W only):")
        print(f"  {traditional}")

        print("\nCDEF (W + θ + MI + τ):")
        print(f"  P(Genuine|Data) = {p_gen:.3f}")
        print(f"  Interpretation  : {interp}")

        collected.append({
            "Scenario": name,
            "Ranking_Type": res.ranking_type,
            "Model": res.distribution_model,
            "W": res.kendalls_W,
            "Theta_scaled": res.theta_scaled,
            "Theta_gumbel": res.theta_gumbel,
            "Avg_tau": res.avg_kendalls_tau,
            "MI": res.mutual_information,
            "Copula_LL": res.avg_log_likelihood,
            "Model_LL": res.model_log_likelihood,
            "Rel_Concordance": res.relative_importance.get("Concordance", 0.0),
            "Rel_Concurrence": res.relative_importance.get("Concurrence", 0.0),
            "Rel_Extremeness": res.relative_importance.get("Extremeness", 0.0),
            "Traditional": traditional,
            "P_Genuine": p_gen,
            "CDEF_Interpretation": interp,
        })

    print("\n" + "=" * 80)
    print("SUMMARY: Traditional vs CDEF Analysis")
    print("=" * 80)

    df_results = pd.DataFrame(collected)

    print("\nTraditional (Kendall's W only):")
    for _, row in df_results.iterrows():
        print(f"  {row['Scenario']:30s}: W={row['W']:.3f} → {row['Traditional']}")

    print("\nCDEF Copula (Full Dependency):")
    for _, row in df_results.iterrows():
        print(f"\n  {row['Scenario']:30s}:")
        print(f"    Type: {row['Ranking_Type']}, Model: {row['Model']}")
        print(f"    W={row['W']:.3f}, θ={row['Theta_scaled']:.3f}, MI={row['MI']:.3f}, τ={row['Avg_tau']:.3f}")
        print(f"    Rel: Conc={row['Rel_Concordance']:.3f}, "
              f"Concur={row['Rel_Concurrence']:.3f}, Extreme={row['Rel_Extremeness']:.3f}")
        print(f"    → P(Genuine|Data) = {row['P_Genuine']:.3f}")
        print(f"    → {row['CDEF_Interpretation']}")

    # Optionally export a CSV summary:
    # df_results.to_csv("cdef_summary_fixed.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

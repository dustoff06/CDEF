# src/cdef_analyzer/demo.py
"""
CDEF Demonstration: Detecting Phantom vs Genuine Concordance

Uses the fixed Gumbel copula analyzer with:
- Auto-detection of forced vs non-forced rankings
- Mallows model for forced rankings under dependence
- Proper log-likelihoods
- Relative importance (not conditional probabilities)
"""

from typing import Tuple
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd

from cdef_analyzer import RankDependencyAnalyzer, CopulaResults


# --------------------------
# Scenario generators
# --------------------------
def create_phantom_scenario() -> pd.DataFrame:
    """
    Phantom: HIGH W + VERY HIGH theta + Shared extreme bias
    All raters share the SAME extreme bias pattern
    """
    n_teams, _ = 136, 4

    # Create identical extreme pattern: alternate top and bottom ranks
    extreme_pattern = []
    for i in range(n_teams // 2):
        extreme_pattern.append(i + 1)              # Top
        extreme_pattern.append(n_teams - i)        # Bottom
    if n_teams % 2 == 1:
        extreme_pattern.append(n_teams // 2 + 1)

    rankings = {}
    rater_names = ['CBS', 'CFN', 'Congrove', 'NYT']
    for rater in rater_names:
        pattern = extreme_pattern.copy()
        # Tiny variation to avoid perfect correlation
        for _ in range(2):
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern

    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def create_genuine_scenario() -> pd.DataFrame:
    """
    Genuine: HIGH W + MODERATE theta + Natural agreement
    """
    n_teams = 136
    base_pattern = list(range(1, n_teams + 1))

    rankings = {}
    rater_names = ['CBS', 'CFN', 'Congrove', 'NYT']
    for rater in rater_names:
        pattern = base_pattern.copy()
        for _ in range(12):  # moderate perturbations
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern

    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def create_random_scenario() -> pd.DataFrame:
    """Random: LOW W + LOW theta + No agreement"""
    n_teams = 136
    rankings = {}
    rater_names = ['CBS', 'CFN', 'Congrove', 'NYT']
    for rater in rater_names:
        rankings[rater] = list(np.random.permutation(range(1, n_teams + 1)))
    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def create_clustered_scenario() -> pd.DataFrame:
    """
    Clustered: HIGH W (among 3) + One divergent rater
    """
    n_teams = 136
    base_pattern = list(range(1, n_teams + 1))

    rankings = {}
    for rater in ['CBS', 'CFN', 'NYT']:
        pattern = base_pattern.copy()
        for _ in range(8):  # tight cluster
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern

    pattern = base_pattern.copy()
    for _ in range(40):  # divergent
        i, j = np.random.choice(n_teams, 2, replace=False)
        pattern[i], pattern[j] = pattern[j], pattern[i]
    rankings['Congrove'] = pattern

    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


# --------------------------
# Interpretation helper
# --------------------------
def cdef_interpretation(results: CopulaResults) -> Tuple[float, str]:
    W = results.kendalls_W
    theta = results.theta_scaled
    mi = results.mutual_information
    avg_tau = results.avg_kendalls_tau

    if W > 0.85 and theta > 15.0:
        return 0.10, "⚠️ PHANTOM: Shared extreme biases (very high W + very high θ)"
    elif W > 0.75 and theta > 8.0:
        return 0.20, "⚠️ Likely PHANTOM: High concordance with extreme tail dependence"
    elif W > 0.65 and 2.5 < theta < 6.5:
        return 0.75, "✓ GENUINE: Natural agreement (high W + moderate θ)"
    elif W > 0.75 and theta < 4.0:
        return 0.85, "✓✓ STRONG GENUINE: Excellent natural agreement"
    elif W > 0.5 and 0.3 < avg_tau < 0.65:
        return 0.60, "→ CLUSTERED: Subgroup agreement with divergent rater(s)"
    elif W > 0.25:
        return 0.70, "◐ WEAK: Limited systematic agreement"
    else:
        return 0.90, "○ RANDOM: No systematic agreement (independence)"


# --------------------------
# I/O helpers
# --------------------------
def save_scenario_to_excel(rankings_df: pd.DataFrame, filepath: Path) -> None:
    """Save scenario in long format matching empirical input schema"""
    long_format = []
    for team_idx in range(len(rankings_df)):
        for rater in rankings_df.columns:
            long_format.append({
                'Ratee': f'Team_{team_idx+1}',
                'Rater': rater,
                'Ranking': int(rankings_df.iloc[team_idx][rater])
            })
    df_long = pd.DataFrame(long_format)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_excel(filepath, sheet_name='Sheet1', index=False)
    print(f"  Saved to: {filepath}")


# --------------------------
# Main demo
# --------------------------
def run_cdef_demonstration(outdir: Path, save_inputs: bool = True, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    print("\n" + "=" * 80)
    print("CDEF DEMONSTRATION: Detecting Phantom vs Genuine Concordance")
    print("Using Properly Fixed Gumbel Copula Analysis")
    print("=" * 80)

    scenarios = {
        'Phantom (Extreme Bias)': create_phantom_scenario(),
        'Genuine (Natural Agreement)': create_genuine_scenario(),
        'Random (No Agreement)': create_random_scenario(),
        'Clustered (Outlier)': create_clustered_scenario()
    }

    results_list = []

    for name, rankings_df in scenarios.items():
        print(f"\n{'=' * 80}")
        print(f"{name.upper()}")
        print('=' * 80)

        # Persist scenario (optional)
        filename = f"scenario_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.xlsx"
        scenario_path = outdir / "scenarios" / filename
        if save_inputs:
            save_scenario_to_excel(rankings_df, scenario_path)

        # Temp file for the analyzer
        temp_file = outdir / "tmp" / f"temp_{filename}"
        save_scenario_to_excel(rankings_df, temp_file)

        analyzer = RankDependencyAnalyzer(random_seed=seed)
        try:
            results = analyzer.analyze_from_excel(
                str(temp_file), 'Sheet1', 'Rater', 'Ratee', 'Ranking'
            )
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            continue

        # CDEF interpretation
        cdef_prob, cdef_interp = cdef_interpretation(results)

        # Traditional interpretation (W only)
        if results.kendalls_W > 0.7:
            traditional = "High concordance → Good agreement"
        elif results.kendalls_W > 0.4:
            traditional = "Moderate concordance → Some agreement"
        else:
            traditional = "Low concordance → Poor agreement"

        # Display
        print(f"\nRanking Type: {results.ranking_type}")
        print(f"Distribution Model: {results.distribution_model}")
        if results.model_log_likelihood is not None:
            print(f"Model Log-Likelihood: {results.model_log_likelihood}")

        print("\nCore Metrics:")
        print(f"  Kendall's W (concordance):     {results.kendalls_W:.3f}")
        print(f"  Theta (scaled):                {results.theta_scaled:.3f}")
        print(f"  Gumbel theta (from tau):       {results.theta_gumbel:.3f}")
        print(f"  Avg Kendall's tau:             {results.avg_kendalls_tau:.3f}")
        print(f"  Mutual information:            {results.mutual_information:.3f}")

        print("\nLog-Likelihoods:")
        print(f"  Copula (average):              {results.avg_log_likelihood:.6f}")
        print(f"  Independence baseline:         {results.independence_log_likelihood:.6f}")

        print("\nRelative Importance (NOT probabilities):")
        for key, val in results.relative_importance.items():
            print(f"  {key:15s}: {val:.3f}")

        print("\nPairwise Theta Range:")
        theta_values = list(results.pairwise_thetas.values())
        if theta_values:
            print(f"  Max: {max(theta_values):.3f}, Min: {min(theta_values):.3f}")

        print("\nTraditional Analysis (W only):")
        print(f"  {traditional}")

        print("\nCDEF Analysis (W + θ + MI + τ):")
        print(f"  P(Genuine|Data) = {cdef_prob:.3f}")
        print(f"  Interpretation: {cdef_interp}")

        results_list.append({
            'Scenario': name,
            'Ranking_Type': results.ranking_type,
            'Model': results.distribution_model,
            'W': results.kendalls_W,
            'Theta_scaled': results.theta_scaled,
            'Theta_gumbel': results.theta_gumbel,
            'Avg_tau': results.avg_kendalls_tau,
            'MI': results.mutual_information,
            'Copula_LL': results.avg_log_likelihood,
            'Model_LL': results.model_log_likelihood,
            'Rel_Concordance': results.relative_importance['Concordance'],
            'Rel_Concurrence': results.relative_importance['Concurrence'],
            'Rel_Extremeness': results.relative_importance['Extremeness'],
            'Traditional': traditional,
            'P_Genuine': cdef_prob,
            'CDEF_Interpretation': cdef_interp
        })

        # Clean temp
        try:
            temp_file.unlink(missing_ok=True)  # Py3.8+: missing_ok param (if 3.8-, wrap in exists check)
        except TypeError:
            if temp_file.exists():
                temp_file.unlink()

    print("\n" + "=" * 80)
    print("SUMMARY: Traditional vs CDEF Analysis")
    print("=" * 80)
    df_results = pd.DataFrame(results_list)

    print("\nTraditional Analysis (Kendall's W only):")
    for _, row in df_results.iterrows():
        print(f"  {row['Scenario']:30s}: W={row['W']:.3f} → {row['Traditional']}")

    print("\nCDEF Copula Analysis (Full Dependency Structure):")
    for _, row in df_results.iterrows():
        print(f"\n  {row['Scenario']:30s}:")
        print(f"    Type: {row['Ranking_Type']}, Model: {row['Model']}")
        print(f"    W={row['W']:.3f}, θ={row['Theta_scaled']:.3f}, MI={row['MI']:.3f}, τ={row['Avg_tau']:.3f}")
        print(f"    Relative Importance: Conc={row['Rel_Concordance']:.3f}, "
              f"Concur={row['Rel_Concurrence']:.3f}, Extreme={row['Rel_Extremeness']:.3f}")
        print(f"    → P(Genuine|Data) = {row['P_Genuine']:.3f}")
        print(f"    → {row['CDEF_Interpretation']}")

    # Save summary CSV in outdir
    out_csv = outdir / "cdef_summary_fixed.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n✓ Summary saved to: {out_csv}")
    return df_results


def main():
    ap = argparse.ArgumentParser(description="Run CDEF demonstration scenarios.")
    ap.add_argument("--outdir", type=Path, default=Path("./outputs"),
                    help="Output directory for scenario files and summary CSV.")
    ap.add_argument("--no-save-inputs", action="store_true",
                    help="Do not persist scenario Excel files.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    args = ap.parse_args()

    run_cdef_demonstration(outdir=args.outdir,
                           save_inputs=not args.no_save_inputs,
                           seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

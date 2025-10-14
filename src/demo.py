# src/cdef_analyzer/demo.py
"""
CDEF Demonstration / CLI

Two modes:
  1) Scenarios demo (default): generates four stylized scenarios and analyzes them.
  2) Single file: analyze a provided Excel file in long format (Rater/Ratee/Ranking).

Usage examples:
  python -m cdef_analyzer.demo --mode scenarios --outdir outputs --seed 42
  python -m cdef_analyzer.demo --mode file \
    --excel src/examples/data.xlsx --sheet Sheet1 \
    --rater-col Rater --ratee-col Ratee --ranking-col Ranking --seed 42
"""
from __future__ import annotations

from typing import Tuple
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from cdef_analyzer import RankDependencyAnalyzer, CopulaResults


# --------------------------
# Scenario generators
# --------------------------
def create_phantom_scenario() -> pd.DataFrame:
    n_teams = 136
    extreme_pattern = []
    for i in range(n_teams // 2):
        extreme_pattern.append(i + 1)
        extreme_pattern.append(n_teams - i)
    if n_teams % 2 == 1:
        extreme_pattern.append(n_teams // 2 + 1)

    rankings = {}
    rater_names = ['CBS', 'CFN', 'Congrove', 'NYT']
    for rater in rater_names:
        pattern = extreme_pattern.copy()
        for _ in range(2):
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern

    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def create_genuine_scenario() -> pd.DataFrame:
    n_teams = 136
    base_pattern = list(range(1, n_teams + 1))
    rankings = {}
    for rater in ['CBS', 'CFN', 'Congrove', 'NYT']:
        pattern = base_pattern.copy()
        for _ in range(12):
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern
    df = pd.DataFrame(rankings); df.index.name = 'Team'
    return df


def create_random_scenario() -> pd.DataFrame:
    n_teams = 136
    rankings = {}
    for r in ['CBS', 'CFN', 'Congrove', 'NYT']:
        rankings[r] = list(np.random.permutation(range(1, n_teams + 1)))
    df = pd.DataFrame(rankings); df.index.name = 'Team'
    return df


def create_clustered_scenario() -> pd.DataFrame:
    n_teams = 136
    base = list(range(1, n_teams + 1))
    rankings = {}
    for r in ['CBS', 'CFN', 'NYT']:
        p = base.copy()
        for _ in range(8):
            i, j = np.random.choice(n_teams, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        rankings[r] = p
    p = base.copy()
    for _ in range(40):
        i, j = np.random.choice(n_teams, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    rankings['Congrove'] = p
    df = pd.DataFrame(rankings); df.index.name = 'Team'
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
# I/O helper
# --------------------------
def save_scenario_to_excel(rankings_df: pd.DataFrame, filepath: Path) -> None:
    long_format = []
    for i in range(len(rankings_df)):
        for rater in rankings_df.columns:
            long_format.append({
                'Ratee': f'Team_{i+1}',
                'Rater': rater,
                'Ranking': int(rankings_df.iloc[i][rater])
            })
    df_long = pd.DataFrame(long_format)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_excel(filepath, sheet_name='Sheet1', index=False)


# --------------------------
# Run scenarios
# --------------------------
def run_scenarios(outdir: Path, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    scenarios = {
        'Phantom (Extreme Bias)': create_phantom_scenario(),
        'Genuine (Natural Agreement)': create_genuine_scenario(),
        'Random (No Agreement)': create_random_scenario(),
        'Clustered (Outlier)': create_clustered_scenario()
    }

    print("\n" + "=" * 80)
    print("CDEF DEMONSTRATION (scenarios)")
    print("=" * 80)

    results_list = []
    for name, rankings_df in scenarios.items():
        print(f"\n{'=' * 80}\n{name.upper()}\n{'=' * 80}")
        temp_file = outdir / "tmp" / f"temp_{name.replace(' ', '_').replace('(', '').replace(')', '')}.xlsx"
        save_scenario_to_excel(rankings_df, temp_file)

        analyzer = RankDependencyAnalyzer(random_seed=seed)
        try:
            results = analyzer.analyze_from_excel(str(temp_file), 'Sheet1', 'Rater', 'Ratee', 'Ranking')
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            continue

        p_gen, interp = cdef_interpretation(results)
        traditional = ("High concordance → Good agreement" if results.kendalls_W > 0.7
                       else "Moderate concordance → Some agreement" if results.kendalls_W > 0.4
                       else "Low concordance → Poor agreement")

        print(f"\nRanking Type: {results.ranking_type}")
        print(f"Distribution Model: {results.distribution_model}")
        if results.model_log_likelihood is not None:
            print(f"Model Log-Likelihood: {results.model_log_likelihood}")

        print("\nCore Metrics:")
        print(f"  Kendall's W:                  {results.kendalls_W:.3f}")
        print(f"  θ (scaled):                   {results.theta_scaled:.3f}")
        print(f"  θ (Gumbel from τ):            {results.theta_gumbel:.3f}")
        print(f"  Avg Kendall's τ:              {results.avg_kendalls_tau:.3f}")
        print(f"  Mutual information:           {results.mutual_information:.3f}")

        print("\nLog-Likelihoods:")
        print(f"  Copula (average):             {results.avg_log_likelihood:.6f}")
        print(f"  Independence baseline:        {results.independence_log_likelihood:.6f}")

        print("\nRelative Importance (NOT probabilities):")
        for k, v in results.relative_importance.items():
            print(f"  {k:15s}: {v:.3f}")

        if results.pairwise_thetas:
            ths = list(results.pairwise_thetas.values())
            print("\nPairwise Theta Range:")
            print(f"  Max: {max(ths):.3f}, Min: {min(ths):.3f}")

        print("\nTraditional Analysis (W only):")
        print(f"  {traditional}")

        print("\nCDEF Analysis (W + θ + MI + τ):")
        print(f"  P(Genuine|Data) = {p_gen:.3f}")
        print(f"  Interpretation: {interp}")

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
            'P_Genuine': p_gen,
            'CDEF_Interpretation': interp
        })

    df_results = pd.DataFrame(results_list)
    out_csv = outdir / "cdef_summary_fixed.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n✓ Summary saved to: {out_csv}")
    return df_results


# --------------------------
# Run single file
# --------------------------
def run_file(excel: Path, sheet: str, rater_col: str, ratee_col: str, ranking_col: str, seed: int) -> None:
    np.random.seed(seed)
    analyzer = RankDependencyAnalyzer(random_seed=seed)
    results = analyzer.analyze_from_excel(str(excel), sheet, rater_col, ratee_col, ranking_col)

    p_gen, interp = cdef_interpretation(results)

    print("\n" + "=" * 80)
    print("CDEF ANALYSIS (single file)")
    print("=" * 80)
    print(f"Ranking Type: {results.ranking_type}")
    print(f"Distribution Model: {results.distribution_model}")
    if results.model_log_likelihood is not None:
        print(f"Model Log-Likelihood: {results.model_log_likelihood}")

    print("\nCore Metrics:")
    print(f"  Kendall's W:                  {results.kendalls_W:.3f}")
    print(f"  θ (scaled):                   {results.theta_scaled:.3f}")
    print(f"  θ (Gumbel from τ):            {results.theta_gumbel:.3f}")
    print(f"  Avg Kendall's τ:              {results.avg_kendalls_tau:.3f}")
    print(f"  Mutual information:           {results.mutual_information:.3f}")

    print("\nLog-Likelihoods:")
    print(f"  Copula (average):             {results.avg_log_likelihood:.6f}")
    print(f"  Independence baseline:        {results.independence_log_likelihood:.6f}")

    print("\nRelative Importance (NOT probabilities):")
    for k, v in results.relative_importance.items():
        print(f"  {k:15s}: {v:.3f}")

    if results.pairwise_thetas:
        ths = list(results.pairwise_thetas.values())
        print("\nPairwise Theta Range:")
        print(f"  Max: {max(ths):.3f}, Min: {min(ths):.3f}")

    print("\nCDEF Analysis (W + θ + MI + τ):")
    print(f"  P(Genuine|Data) = {p_gen:.3f}")
    print(f"  Interpretation: {interp}")


# --------------------------
# CLI
# --------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="CDEF demo / CLI")
    ap.add_argument("--mode", choices=["scenarios", "file"], default="scenarios",
                    help="Run stylized scenarios (default) or analyze a single Excel file.")
    ap.add_argument("--outdir", type=Path, default=Path("./outputs"),
                    help="Output directory for scenario summary.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")

    # file mode args
    ap.add_argument("--excel", type=Path, help="Path to Excel file (long format).")
    ap.add_argument("--sheet", type=str, default="Sheet1", help="Excel sheet name.")
    ap.add_argument("--rater-col", type=str, default="Rater", help="Rater column.")
    ap.add_argument("--ratee-col", type=str, default="Ratee", help="Ratee column.")
    ap.add_argument("--ranking-col", type=str, default="Ranking", help="Ranking column.")

    args = ap.parse_args()

    if args.mode == "file":
        if not args.excel:
            ap.error("--excel is required when --mode file")
        run_file(args.excel, args.sheet, args.rater_col, args.ratee_col, args.ranking_col, args.seed)
    else:
        run_scenarios(args.outdir, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

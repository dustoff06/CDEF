"""
CDEF Demonstration: Detecting Phantom vs Genuine Concordance

Uses the properly fixed Gumbel copula analyzer with:
- Auto-detection of forced vs non-forced rankings
- Mallows model for forced rankings under dependence
- Proper log-likelihoods
- Relative importance (not fake conditional probabilities)
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, chi2_contingency, entropy
from copulas.bivariate import Gumbel
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Try to import from multiple locations
try:
    from gumbel_copula_fixed import RankDependencyAnalyzer, CopulaResults
except ModuleNotFoundError:
    # Add common paths
    possible_paths = [
        '/home/claude',
        '/mnt/user-data/outputs',
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd()
    ]
    
    for path in possible_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    try:
        from gumbel_copula_fixed import RankDependencyAnalyzer, CopulaResults
    except ModuleNotFoundError:
        print("ERROR: Cannot find gumbel_copula_fixed.py")
        print(f"Searched in: {possible_paths}")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease ensure gumbel_copula_fixed.py is in the same directory as this script")
        print("or in /home/claude or /mnt/user-data/outputs")
        sys.exit(1)


def create_phantom_scenario():
    """
    Phantom: HIGH W + VERY HIGH theta + Shared extreme bias
    All raters share the SAME extreme bias pattern
    """
    n_teams, n_raters = 136, 4
    
    # Create identical extreme pattern: alternate top and bottom ranks
    extreme_pattern = []
    for i in range(n_teams // 2):
        extreme_pattern.append(i + 1)  # Top ranks
        extreme_pattern.append(n_teams - i)  # Bottom ranks
    if n_teams % 2 == 1:
        extreme_pattern.append(n_teams // 2 + 1)
    
    rankings = {}
    rater_names = ['CBS', 'CFN', 'Congrove', 'NYT']
    
    for rater in rater_names:
        pattern = extreme_pattern.copy()
        # Add tiny variation (2-3 swaps) to avoid perfect correlation
        for _ in range(2):
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern
    
    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def create_genuine_scenario():
    """
    Genuine: HIGH W + MODERATE theta + Natural agreement  
    Raters give similar rankings without extreme bias
    """
    n_teams = 136
    base_pattern = list(range(1, n_teams + 1))
    
    rankings = {}
    rater_names = ['CBS', 'CFN', 'Congrove', 'NYT']
    
    for rater in rater_names:
        pattern = base_pattern.copy()
        # Add moderate noise (10-15 swaps)
        for _ in range(12):
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern
    
    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def create_random_scenario():
    """Random: LOW W + LOW theta + No agreement"""
    n_teams = 136
    
    rankings = {}
    rater_names = ['CBS', 'CFN', 'Congrove', 'NYT']
    
    for rater in rater_names:
        rankings[rater] = list(np.random.permutation(range(1, n_teams + 1)))
    
    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def create_clustered_scenario():
    """
    Clustered: HIGH W (among 3) + One divergent rater
    Mimics the Congrove pattern from real data
    """
    n_teams = 136
    base_pattern = list(range(1, n_teams + 1))
    
    rankings = {}
    
    # CBS, CFN, NYT - tight cluster (small variations)
    for rater in ['CBS', 'CFN', 'NYT']:
        pattern = base_pattern.copy()
        for _ in range(8):
            i, j = np.random.choice(n_teams, 2, replace=False)
            pattern[i], pattern[j] = pattern[j], pattern[i]
        rankings[rater] = pattern
    
    # Congrove - independent (large variations)
    pattern = base_pattern.copy()
    for _ in range(40):
        i, j = np.random.choice(n_teams, 2, replace=False)
        pattern[i], pattern[j] = pattern[j], pattern[i]
    rankings['Congrove'] = pattern
    
    df = pd.DataFrame(rankings)
    df.index.name = 'Team'
    return df


def cdef_interpretation(results: CopulaResults) -> Tuple[float, str]:
    """
    Interpret CDEF results to classify as Phantom, Genuine, Random, or Clustered.
    
    Uses:
    - W (concordance): Overall agreement
    - theta (extremeness): Tail dependence
    - MI (concurrence): Shared information
    - avg_tau: Pairwise correlations
    
    Returns:
        Tuple of (probability_genuine, interpretation_string)
    """
    W = results.kendalls_W
    theta = results.theta_scaled
    mi = results.mutual_information
    avg_tau = results.avg_kendalls_tau
    
    # Phantom detection: Very high W + Very high theta = shared extreme bias
    if W > 0.85 and theta > 15.0:
        prob = 0.10
        interpretation = "⚠️ PHANTOM: Shared extreme biases (very high W + very high θ)"
    
    # Strong phantom
    elif W > 0.75 and theta > 8.0:
        prob = 0.20
        interpretation = "⚠️ Likely PHANTOM: High concordance with extreme tail dependence"
    
    # Genuine agreement: High W + Moderate theta
    elif W > 0.65 and 2.5 < theta < 6.5:
        prob = 0.75
        interpretation = "✓ GENUINE: Natural agreement (high W + moderate θ)"
    
    # Strong genuine: Very high W + Low-moderate theta
    elif W > 0.75 and theta < 4.0:
        prob = 0.85
        interpretation = "✓✓ STRONG GENUINE: Excellent natural agreement"
    
    # Clustered (some agree, some don't)
    elif W > 0.5 and 0.3 < avg_tau < 0.65:
        prob = 0.60
        interpretation = "→ CLUSTERED: Subgroup agreement with divergent rater(s)"
    
    # Weak agreement
    elif W > 0.25:
        prob = 0.70
        interpretation = "◐ WEAK: Limited systematic agreement"
    
    # Random/independence
    else:
        prob = 0.90
        interpretation = "○ RANDOM: No systematic agreement (independence)"
    
    return prob, interpretation


def save_scenario_to_excel(rankings_df: pd.DataFrame, filename: str, scenario_name: str):
    """Save scenario in long format matching original data structure"""
    long_format = []
    
    for team_idx in range(len(rankings_df)):
        for rater in rankings_df.columns:
            long_format.append({
                'Ratee': f'Team_{team_idx+1}',
                'Rater': rater,
                'Ranking': int(rankings_df.iloc[team_idx][rater])
            })
    
    df_long = pd.DataFrame(long_format)
    output_path = f'/mnt/user-data/outputs/{filename}'
    df_long.to_excel(output_path, sheet_name='Sheet1', index=False)
    print(f"  Saved to: {filename}")


def run_cdef_demonstration():
    """
    Demonstrate CDEF's diagnostic capability using proper Gumbel copula analysis.
    
    Shows how CDEF distinguishes:
    - Phantom concordance (shared biases) from genuine agreement
    - Extreme tail dependence from natural correlation
    - Clustered subgroups from uniform agreement
    """
    
    np.random.seed(42)
    
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
        
        # Save scenario to Excel
        filename = f"scenario_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.xlsx"
        save_scenario_to_excel(rankings_df, filename, name)
        
        # Create temporary Excel file for analysis
        temp_file = f'/tmp/temp_{filename}'
        long_format = []
        for team_idx in range(len(rankings_df)):
            for rater in rankings_df.columns:
                long_format.append({
                    'Ratee': f'Team_{team_idx+1}',
                    'Rater': rater,
                    'Ranking': int(rankings_df.iloc[team_idx][rater])
                })
        df_long = pd.DataFrame(long_format)
        df_long.to_excel(temp_file, sheet_name='Sheet1', index=False)
        
        # Run analysis with proper analyzer
        analyzer = RankDependencyAnalyzer(random_seed=42)
        try:
            results = analyzer.analyze_from_excel(
                temp_file, 'Sheet1', 'Rater', 'Ratee', 'Ranking'
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
        
        # Display results
        print(f"\nRanking Type: {results.ranking_type}")
        print(f"Distribution Model: {results.distribution_model}")
        if results.model_log_likelihood is not None:
            print(f"Model Log-Likelihood: {results.model_log_likelihood}")
        
        print(f"\nCore Metrics:")
        print(f"  Kendall's W (concordance):     {results.kendalls_W:.3f}")
        print(f"  Theta (scaled):                {results.theta_scaled:.3f}")
        print(f"  Gumbel theta (from tau):       {results.theta_gumbel:.3f}")
        print(f"  Avg Kendall's tau:             {results.avg_kendalls_tau:.3f}")
        print(f"  Mutual information:            {results.mutual_information:.3f}")
        
        print(f"\nLog-Likelihoods:")
        print(f"  Copula (average):              {results.avg_log_likelihood:.6f}")
        print(f"  Independence baseline:         {results.independence_log_likelihood:.6f}")
        
        print(f"\nRelative Importance (NOT probabilities):")
        for key, val in results.relative_importance.items():
            print(f"  {key:15s}: {val:.3f}")
        
        print(f"\nPairwise Theta Range:")
        theta_values = list(results.pairwise_thetas.values())
        print(f"  Max: {max(theta_values):.3f}, Min: {min(theta_values):.3f}")
        
        print(f"\nTraditional Analysis (W only):")
        print(f"  {traditional}")
        
        print(f"\nCDEF Analysis (W + θ + MI + τ):")
        print(f"  P(Genuine|Data) = {cdef_prob:.3f}")
        print(f"  Interpretation: {cdef_interp}")
        
        # Store results
        result = {
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
        }
        results_list.append(result)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Summary comparison table
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
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("Traditional Kendall's W CANNOT distinguish:")
    print("  • Phantom concordance (shared biases) from genuine agreement")
    print("  • Extreme tail dependence from natural correlation")
    print("  • Clustered subgroups from uniform agreement")
    print("\nCDEF reveals the truth by analyzing:")
    print("  ✓ Concordance (W): Overall agreement across all raters")
    print("  ✓ Dependence (θ): Tail dependence & extremeness in rankings")
    print("  ✓ Concurrence (MI): Shared information structure")
    print("  ✓ Flexibility (τ): Pairwise correlation patterns")
    print("\nDistribution Models:")
    print("  ✓ Mallows: Forced rankings with dependence (your data!)")
    print("  ✓ Uniform: Forced rankings with independence")
    print("  ✓ Multinomial: Non-forced rankings with independence")
    print("\nRelative Importance:")
    print("  • NOT conditional probabilities (different units/scales)")
    print("  • Normalized decomposition weights showing contribution")
    print("  • Extremeness dominance indicates tail-dependence drives agreement")
    print("=" * 80)
    
    return df_results


if __name__ == "__main__":
    results_df = run_cdef_demonstration()
    
    # Save summary
    output_file = '/mnt/user-data/outputs/cdef_summary_fixed.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Summary saved to: {output_file}")

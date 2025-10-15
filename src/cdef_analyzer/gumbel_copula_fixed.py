"""
Gumbel Copula Analyzer for Ranking Data

Handles both forced-choice rankings (strict permutations) and non-forced rankings (ties allowed).
Auto-detects ranking type and applies appropriate statistical models.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, chi2_contingency, entropy, multinomial
from scipy.spatial.distance import cdist
from copulas.bivariate import Gumbel
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
import math


@dataclass
class CopulaResults:
    """Container for copula analysis results"""
    theta_scaled: float
    theta_gumbel: float
    kendalls_W: float
    avg_kendalls_tau: float
    mutual_information: float
    chi_square_stat: float
    p_value: float
    avg_log_likelihood: float
    independence_log_likelihood: float
    pairwise_thetas: Dict[str, float]
    tau_matrix: pd.DataFrame
    ranking_type: str
    distribution_model: str
    model_log_likelihood: Optional[float]
    relative_importance: Dict[str, float]
    n_raters: int
    n_items: int


class RankDependencyAnalyzer:
    """
    Analyze ranking dependence using Gumbel copulas.
    
    Features:
    - Auto-detects forced vs non-forced rankings
    - Applies appropriate statistical models (Mallows, Multinomial, etc.)
    - Computes dependence via Gumbel copulas (upper-tail dependence)
    - Provides interpretable metrics (Kendall's W, tau, theta)
    """
    
    def __init__(self, num_samples: int = 10000, significance_level: float = 0.05, 
                 random_seed: Optional[int] = 42):
        """
        Initialize analyzer.
        
        Args:
            num_samples: Number of Monte Carlo samples for permutation tests
            significance_level: Alpha level for hypothesis tests
            random_seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.significance_level = significance_level
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.copulas: Dict[str, Gumbel] = {}
        self.theta: Optional[float] = None
        self.W: Optional[float] = None
        
    def load_excel(self, file_path: str, sheet_name: str, 
                   rater_col: str, ratee_col: str, ranking_col: str) -> pd.DataFrame:
        """
        Load rankings from Excel in long format, convert to wide format.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name
            rater_col: Column name for raters
            ratee_col: Column name for items being ranked
            ranking_col: Column name for rank values
            
        Returns:
            DataFrame with ratees as rows, raters as columns
            
        Raises:
            ValueError: If required columns are missing
            FileNotFoundError: If file doesn't exist
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Excel file not found: {file_path}") from e
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}") from e
        
        # Validate columns
        required_cols = [rater_col, ratee_col, ranking_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Pivot to wide format
        rankings_wide = df.pivot(index=ratee_col, columns=rater_col, values=ranking_col)
        
        # Drop rows with missing values
        n_before = len(rankings_wide)
        rankings_wide = rankings_wide.dropna()
        n_after = len(rankings_wide)
        
        if n_after < n_before:
            warnings.warn(f"Dropped {n_before - n_after} rows with missing values")
        
        if n_after == 0:
            raise ValueError("No complete rankings found after dropping missing values")
        
        return rankings_wide.astype(int)
    
    def detect_ranking_type(self, rankings_df: pd.DataFrame) -> str:
        """
        Detect if rankings are forced-choice (permutation) or non-forced (ties allowed).
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            'forced' if strict permutations, 'non-forced' if ties exist
        """
        n_items = len(rankings_df)
        
        for col in rankings_df.columns:
            ranks = rankings_df[col]
            
            # Check for duplicate ranks (ties)
            if len(ranks) != len(ranks.unique()):
                return 'non-forced'
            
            # Check if it's a complete permutation [1, 2, ..., n]
            expected_ranks = set(range(1, n_items + 1))
            if set(ranks) != expected_ranks:
                return 'non-forced'
        
        return 'forced'
    
    def compute_kendalls_W(self, rankings_df: pd.DataFrame) -> float:
        """
        Compute Kendall's W (coefficient of concordance).
        
        W = 1 indicates perfect agreement, W = 0 indicates no agreement.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Kendall's W
        """
        N, m = rankings_df.shape  # N items, m raters
        row_sums = rankings_df.sum(axis=1)
        mean_rank_sum = row_sums.mean()
        S = np.sum((row_sums - mean_rank_sum) ** 2)
        W = (12 * S) / (m ** 2 * (N ** 3 - N))
        return round(W, 3)
    
    def compute_mutual_information_and_independence(
        self, rankings1: np.ndarray, rankings2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute mutual information and chi-square test for independence.
    
        Key fix: drop any all-zero rows/columns from the contingency table
        before running chi-square to avoid "expected frequency == 0" errors.
        If the table collapses below 2x2, return (mi, p=1.0, chi2=0.0).
    
        Args:
            rankings1: First rater's rankings
            rankings2: Second rater's rankings
    
        Returns:
            Tuple of (mutual_information, p_value, chi_square_statistic)
        """
        n = len(rankings1)
        # bins: at least 5 for chi-square validity, but never more than n//2 to avoid sparsity
        bins = max(5, min(int(np.ceil(np.sqrt(n))), max(5, n // 2)))
    
        # 2D histogram (contingency table)
        joint_dist, _, _ = np.histogram2d(
            rankings1, rankings2,
            bins=bins,
            range=[[rankings1.min(), rankings1.max()],
                   [rankings2.min(), rankings2.max()]]
        )
    
        # ---- SURGICAL FIX: drop all-zero rows/columns before chi-square ----
        row_mask = joint_dist.sum(axis=1) > 0
        col_mask = joint_dist.sum(axis=0) > 0
        reduced = joint_dist[np.ix_(row_mask, col_mask)]
    
        # If the reduced table is too small, skip chi-square safely
        if reduced.size == 0 or reduced.shape[0] < 2 or reduced.shape[1] < 2:
            # still compute MI on the smoothed full table (below), but report no evidence of dependence
            chi2, p_value = 0.0, 1.0
        else:
            # Disable Yates continuity correction for stability in sparse-but-valid tables
            chi2, p_value, _, _ = chi2_contingency(reduced, correction=False)
            chi2 = float(chi2)
            p_value = float(p_value)
        # -------------------------------------------------------------------
    
        # Mutual information on a smoothed version of the original table (numerically stable)
        joint_smooth = joint_dist + 1e-12
        joint_norm = joint_smooth / np.sum(joint_smooth)
        mx = np.sum(joint_norm, axis=1)
        my = np.sum(joint_norm, axis=0)
        mi = (entropy(mx) + entropy(my) - entropy(joint_norm.flatten()))
        mi = float(np.round(mi, 3))
    
        return mi, round(p_value, 3), round(chi2, 3)

    
    def estimate_gumbel_theta(self, rankings_df: pd.DataFrame) -> float:
        """
        Estimate Gumbel copula parameter from Kendall's tau.
        
        For Gumbel copula: θ = 1/(1-τ), where τ is Kendall's tau.
        θ ≥ 1, with θ=1 indicating independence.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Gumbel theta parameter
        """
        raters = rankings_df.columns
        taus = []
        
        for i, r1 in enumerate(raters):
            for r2 in raters[i + 1:]:
                tau, _ = kendalltau(rankings_df[r1], rankings_df[r2])
                taus.append(tau)
        
        avg_tau = np.mean(taus) if taus else 0
        
        # Gumbel constraint: theta >= 1
        if avg_tau >= 0.99:
            theta = 100.0  # Cap at high value
        elif avg_tau <= 0:
            theta = 1.0  # Independence
        else:
            theta = 1.0 / (1.0 - avg_tau)
        
        return round(theta, 3)
    
    def estimate_copula_theta(self, rankings_df: pd.DataFrame) -> float:
        """
        Estimate scaled theta incorporating both pairwise (tau) and global (W) agreement.
        
        This combines local pairwise dependence with global concordance.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Scaled theta parameter
        """
        theta_gumbel = self.estimate_gumbel_theta(rankings_df)
        self.W = self.compute_kendalls_W(rankings_df)
        
        # Scale theta by (1 + W) to incorporate global concordance
        self.theta = round(theta_gumbel * (1 + self.W), 3)
        return self.theta
    
    def fit_gumbel_copulas(self, rankings_df: pd.DataFrame) -> None:
        """
        Fit bivariate Gumbel copulas for all rater pairs.
        
        Transforms rankings to uniform [0,1] marginals via empirical CDF,
        then fits Gumbel copula to capture upper-tail dependence.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
        """
        raters = rankings_df.columns
        n = len(rankings_df)
        
        for i, r1 in enumerate(raters):
            for r2 in raters[i+1:]:
                try:
                    copula = Gumbel()
                    
                    # Transform to uniform [0,1] using empirical CDF
                    u1 = (rankings_df[r1].rank() - 0.5) / n
                    u2 = (rankings_df[r2].rank() - 0.5) / n
                    data = np.column_stack([u1.values, u2.values])
                    
                    copula.fit(data)
                    self.copulas[f"{r1}-{r2}"] = copula
                    
                except ValueError as e:
                    # Gumbel requires theta >= 1 (positive dependence only)
                    # If fit fails, use independence (theta=1)
                    warnings.warn(
                        f"Could not fit Gumbel copula for {r1}-{r2}: {e}. "
                        f"Using independence (θ=1)."
                    )
                    copula = Gumbel()
                    copula.theta = 1.0
                    self.copulas[f"{r1}-{r2}"] = copula
    
    def compute_avg_log_likelihood(self, rankings_df: pd.DataFrame) -> float:
        """
        Compute average log-likelihood across all observations under fitted copulas.
        
        This properly uses all data points, not just the mean.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Average log-likelihood per observation
        """
        if not self.copulas:
            return 0.0
        
        n = len(rankings_df)
        log_likelihoods = []
        
        for pair_name, copula in self.copulas.items():
            r1, r2 = pair_name.split('-')
            
            # Transform to uniform [0,1]
            u1 = (rankings_df[r1].rank() - 0.5) / n
            u2 = (rankings_df[r2].rank() - 0.5) / n
            
            # Compute log-likelihood for each observation
            for i in range(n):
                point = np.array([[u1.iloc[i], u2.iloc[i]]])
                try:
                    density = copula.probability_density(point)
                    if density > 0:
                        log_likelihoods.append(np.log(density))
                except (ValueError, RuntimeError):
                    continue
        
        if not log_likelihoods:
            return 0.0
        
        return round(np.mean(log_likelihoods), 6)
    
    def compute_independence_log_likelihood(self, rankings_df: pd.DataFrame) -> float:
        """
        Compute log-likelihood under independence (uniform copula).
        
        Under independence, copula density = 1 everywhere, so log-likelihood = 0.
        
        Returns:
            0.0 (independence baseline)
        """
        return 0.0
    
    def compute_consensus_ranking(self, rankings_df: pd.DataFrame) -> pd.Series:
        """
        Compute consensus ranking using Borda count.
        
        Lower sum of ranks = better overall ranking.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Consensus ranking (1 = best)
        """
        borda_scores = rankings_df.sum(axis=1)
        consensus = borda_scores.rank(method='min').astype(int)
        return consensus
    
    def compute_kendall_distance(self, rank1: np.ndarray, rank2: np.ndarray) -> int:
        """
        Compute Kendall tau distance (number of pairwise disagreements).
        
        Args:
            rank1: First ranking
            rank2: Second ranking
            
        Returns:
            Number of discordant pairs
        """
        n = len(rank1)
        distance = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if pair (i,j) is ordered differently in the two rankings
                if (rank1[i] < rank1[j]) != (rank2[i] < rank2[j]):
                    distance += 1
        
        return distance
    
    def fit_mallows_model(self, rankings_df: pd.DataFrame) -> Dict:
        """
        Fit Mallows model for forced rankings under dependence.
        
        Mallows model: P(σ) ∝ exp(-θ * d(σ, σ₀))
        where d is Kendall distance and σ₀ is consensus ranking.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Dict with 'theta' (dispersion), 'consensus' ranking, and log-likelihood
        """
        # Compute consensus ranking
        consensus = self.compute_consensus_ranking(rankings_df)
        
        # Compute average Kendall distance to consensus
        distances = []
        for col in rankings_df.columns:
            d = self.compute_kendall_distance(
                rankings_df[col].values,
                consensus.values
            )
            distances.append(d)
        
        avg_distance = np.mean(distances)
        n = len(rankings_df)
        
        # Estimate theta (simple MLE approximation)
        # For Mallows: E[d] ≈ n(n-1)/(4 * (1 + exp(θ)))
        # Rough inverse: theta ≈ log((n(n-1))/(4*avg_distance) - 1)
        if avg_distance > 0:
            theta_mallows = max(0.01, np.log(max(n * (n-1) / (4 * avg_distance) - 1, 1.01)))
        else:
            theta_mallows = 10.0  # Perfect agreement
        
        # Compute log-likelihood (approximate)
        # L = -θ * Σd(σᵢ, σ₀) - log(Z(θ))
        # Z(θ) is partition function (expensive to compute exactly, use approximation)
        log_likelihood = -theta_mallows * sum(distances)
        
        return {
            'theta': round(theta_mallows, 3),
            'consensus': consensus,
            'avg_distance': round(avg_distance, 2),
            'log_likelihood': round(log_likelihood, 3)
        }
    
    def calculate_multinomial_log_likelihood(self, rankings_df: pd.DataFrame) -> float:
        """
        Compute log-likelihood under multinomial model (independence baseline).
        
        Treats each (team, rank) pair as independent multinomial draw.
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Log-likelihood under multinomial model
        """
        # Flatten to get all (team, rank) pairs
        all_pairs = []
        for col in rankings_df.columns:
            for idx, rank in rankings_df[col].items():
                all_pairs.append((idx, rank))
        
        # Count frequencies
        unique_pairs, counts = np.unique(all_pairs, axis=0, return_counts=True)
        
        # Estimate probabilities (MLE)
        n_total = len(all_pairs)
        probs = counts / n_total
        
        # Multinomial log-likelihood
        log_likelihood = np.sum(counts * np.log(probs + 1e-10))
        
        return round(log_likelihood, 3)
    
    def choose_distribution_model(self, rankings_df: pd.DataFrame) -> Tuple[str, Optional[float]]:
        """
        Choose appropriate distribution based on ranking type and dependence test.
        
        Logic:
        - Forced + Dependent → Mallows model
        - Forced + Independent → Uniform permutation
        - Non-forced + Dependent → Report dependence (use copula)
        - Non-forced + Independent → Multinomial
        
        Args:
            rankings_df: Rankings with ratees as rows, raters as columns
            
        Returns:
            Tuple of (model_name, log_likelihood)
        """
        ranking_type = self.detect_ranking_type(rankings_df)
        
        # Test for dependence
        col1, col2 = rankings_df.columns[0], rankings_df.columns[1]
        _, p_value, _ = self.compute_mutual_information_and_independence(
            rankings_df[col1].values, rankings_df[col2].values
        )
        
        is_dependent = p_value < self.significance_level
        
        if ranking_type == 'forced':
            if is_dependent:
                model_result = self.fit_mallows_model(rankings_df)
                return 'Mallows (forced, dependent)', model_result['log_likelihood']
            else:
                # Uniform over all permutations
                # For large n, use Stirling approximation: log(n!) ≈ n*log(n) - n
                n = len(rankings_df)
                if n > 170:  # factorial(171) overflows
                    log_factorial_n = n * math.log(n) - n  # Stirling
                else:
                    log_factorial_n = math.log(math.factorial(n))
                log_likelihood = -log_factorial_n  # log(1/n!)
                return 'Uniform Permutation (forced, independent)', log_likelihood
        else:  # non-forced
            if is_dependent:
                # Just report that dependence detected, rely on copula
                return 'Dependent (non-forced, use copula)', None
            else:
                log_likelihood = self.calculate_multinomial_log_likelihood(rankings_df)
                return 'Multinomial (non-forced, independent)', log_likelihood
    
    def compute_relative_importance(self, W: float, mi: float, theta: float) -> Dict[str, float]:
        """
        Compute relative importance of three factors: Concordance, Concurrence, Extremeness.
        
        These are NOT conditional probabilities - they are normalized importance weights
        showing the relative contribution of each factor.
        
        Args:
            W: Kendall's W (concordance)
            mi: Mutual information (concurrence)
            theta: Copula parameter (extremeness/tail dependence)
            
        Returns:
            Dict with relative importance weights (sum to 1.0)
        """
        total = W + mi + theta
        
        if total == 0:
            return {
                'Concordance': 0.333,
                'Concurrence': 0.333,
                'Extremeness': 0.333
            }
        
        return {
            'Concordance': round(W / total, 3),
            'Concurrence': round(mi / total, 3),
            'Extremeness': round(theta / total, 3)
        }
    
    def analyze_from_excel(
        self, file_path: str, sheet_name: str, 
        rater_col: str, ratee_col: str, ranking_col: str
    ) -> CopulaResults:
        """
        Complete analysis pipeline from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name
            rater_col: Column name for raters
            ratee_col: Column name for items being ranked
            ranking_col: Column name for rank values
            
        Returns:
            CopulaResults dataclass with all analysis outputs
        """
        # Load data
        rankings_wide = self.load_excel(file_path, sheet_name, rater_col, ratee_col, ranking_col)
        
        print(f"\nLoaded data: {len(rankings_wide)} ratees x {len(rankings_wide.columns)} raters")
        print(f"Raters: {list(rankings_wide.columns)}")
        
        # Detect ranking type
        ranking_type = self.detect_ranking_type(rankings_wide)
        print(f"Ranking type: {ranking_type}")
        
        # Fit Gumbel copulas (pairwise)
        self.fit_gumbel_copulas(rankings_wide)
        
        # Calculate core metrics
        theta_scaled = self.estimate_copula_theta(rankings_wide)
        theta_gumbel = self.estimate_gumbel_theta(rankings_wide)
        
        # Mutual information and chi-square test
        col1, col2 = rankings_wide.columns[0], rankings_wide.columns[1]
        mi, p_value, chi2_stat = self.compute_mutual_information_and_independence(
            rankings_wide[col1].values, rankings_wide[col2].values
        )
        
        # Distribution model selection
        distribution_model, model_log_likelihood = self.choose_distribution_model(rankings_wide)
        
        # Average log-likelihood from copulas
        avg_log_likelihood = self.compute_avg_log_likelihood(rankings_wide)
        
        # Independence baseline
        independence_log_likelihood = self.compute_independence_log_likelihood(rankings_wide)
        
        # Relative importance (not conditional probabilities)
        relative_importance = self.compute_relative_importance(self.W, mi, theta_scaled)
        
        # Tau statistics
        taus = []
        for i, c1 in enumerate(rankings_wide.columns):
            for c2 in rankings_wide.columns[i+1:]:
                tau, _ = kendalltau(rankings_wide[c1], rankings_wide[c2])
                taus.append(tau)
        avg_tau = np.mean(taus) if taus else 0
        
        # Pairwise Gumbel thetas
        pairwise_thetas = {}
        for pair_name, copula in self.copulas.items():
            pairwise_thetas[pair_name] = round(copula.theta, 3)
        
        # Tau matrix
        tau_matrix = pd.DataFrame(
            index=rankings_wide.columns, 
            columns=rankings_wide.columns,
            dtype=float
        )
        for c1 in rankings_wide.columns:
            for c2 in rankings_wide.columns:
                if c1 == c2:
                    tau_matrix.loc[c1, c2] = 1.0
                else:
                    tau, _ = kendalltau(rankings_wide[c1], rankings_wide[c2])
                    tau_matrix.loc[c1, c2] = round(tau, 3)
        
        return CopulaResults(
            theta_scaled=theta_scaled,
            theta_gumbel=theta_gumbel,
            kendalls_W=self.W,
            avg_kendalls_tau=round(avg_tau, 3),
            mutual_information=mi,
            chi_square_stat=chi2_stat,
            p_value=p_value,
            avg_log_likelihood=avg_log_likelihood,
            independence_log_likelihood=independence_log_likelihood,
            pairwise_thetas=pairwise_thetas,
            tau_matrix=tau_matrix,
            ranking_type=ranking_type,
            distribution_model=distribution_model,
            model_log_likelihood=model_log_likelihood,
            relative_importance=relative_importance,
            n_raters=len(rankings_wide.columns),
            n_items=len(rankings_wide)
        )


def format_results(results: CopulaResults) -> str:
    """
    Format results for display.
    
    Args:
        results: CopulaResults dataclass
        
    Returns:
        Formatted string for printing
    """
    output = []
    output.append("=" * 70)
    output.append("GUMBEL COPULA ANALYSIS RESULTS")
    output.append("=" * 70)
    
    output.append(f"\nRanking Type: {results.ranking_type}")
    output.append(f"Distribution Model: {results.distribution_model}")
    if results.model_log_likelihood is not None:
        output.append(f"Model Log-Likelihood: {results.model_log_likelihood}")
    
    output.append(f"\nCore Metrics:")
    output.append(f"  Theta (scaled with W):        {results.theta_scaled}")
    output.append(f"  Gumbel theta (from tau):      {results.theta_gumbel}")
    output.append(f"  Kendall's W (concordance):    {results.kendalls_W}")
    output.append(f"  Avg Kendall's tau:            {results.avg_kendalls_tau}")
    
    output.append(f"\nDependence Tests:")
    output.append(f"  Mutual information:           {results.mutual_information}")
    output.append(f"  Chi-square statistic:         {results.chi_square_stat}")
    output.append(f"  p-value:                      {results.p_value}")
    
    output.append(f"\nLog-Likelihoods:")
    output.append(f"  Copula (average):             {results.avg_log_likelihood}")
    output.append(f"  Independence baseline:        {results.independence_log_likelihood}")
    
    output.append(f"\nRelative Importance (NOT probabilities):")
    for key, val in results.relative_importance.items():
        output.append(f"  {key:15s}: {val:.3f}")
    
    output.append(f"\nPairwise Gumbel Thetas:")
    for pair, theta_val in results.pairwise_thetas.items():
        output.append(f"  {pair}: {theta_val}")
    
    output.append(f"\nKendall's Tau Matrix:")
    output.append(str(results.tau_matrix))
    
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    analyzer = RankDependencyAnalyzer(random_seed=42)
    
    file_path = "/mnt/user-data/uploads/data.xlsx"
    
    try:
        results = analyzer.analyze_from_excel(
            file_path, "Sheet1", "Rater", "Ratee", "Ranking"
        )
        
        print(format_results(results))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

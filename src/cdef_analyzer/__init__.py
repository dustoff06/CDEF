"""
CDEF Analyzer: Copula-based Detection of Extremeness & Flexibility

A statistical framework for analyzing ranking concordance using Gumbel copulas,
distinguishing genuine agreement from phantom concordance (shared biases).
"""

from .core import RankDependencyAnalyzer, CopulaResults

__all__ = ["RankDependencyAnalyzer", "CopulaResults"]

from .core import (
    analyze_excel,
    analyze_dataframe,
    results_to_dict,
    run_demo_scenarios,
)

__version__ = "1.0.0"

__all__ = [
    "RankDependencyAnalyzer",
    "CopulaResults",
    "analyze_excel",
    "analyze_dataframe",
    "results_to_dict",
    "run_demo_scenarios",
    "__version__",
]

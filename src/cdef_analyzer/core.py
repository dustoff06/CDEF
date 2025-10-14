# src/cdef_analyzer/core.py
from .gumbel_copula_fixed import (
    RankDependencyAnalyzer as RankDependencyAnalyzer,
    CopulaResults as CopulaResults,
)

__all__ = ["RankDependencyAnalyzer", "CopulaResults"]

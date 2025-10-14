# src/cdef_analyzer/demo.py
from __future__ import annotations
from .gumbel_copula_fixed import RankDependencyAnalyzer, CopulaResults
from .cdef_demo_fixed import run_cdef_demonstration  # if you kept this module
# If you folded everything into gumbel_copula_fixed, just import the function from there

def main() -> int:
    run_cdef_demonstration()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

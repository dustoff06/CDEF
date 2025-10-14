# src/cdef_analyzer/demo.py
"""
Simple CLI demo for the CDEF analyzer.

Usage:
  python -m cdef_analyzer.demo \
    --excel src/examples/data.xlsx \
    --sheet Sheet1 \
    --rater-col Rater --ratee-col Ratee --ranking-col Ranking \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import RankDependencyAnalyzer, CopulaResults  # from gumbel_copula_fixed


def format_summary(res: CopulaResults) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("CDEF / Gumbel Copula — Summary")
    lines.append("=" * 72)
    lines.append(f"Items x Raters           : {res.n_items} x {res.n_raters}")
    lines.append(f"Ranking type             : {res.ranking_type}")
    lines.append(f"Distribution model       : {res.distribution_model}")
    if res.model_log_likelihood is not None:
        lines.append(f"Model log-likelihood     : {res.model_log_likelihood}")
    lines.append("")
    lines.append("Core metrics")
    lines.append(f"  Kendall's W (concord.) : {res.kendalls_W:.3f}")
    lines.append(f"  Avg Kendall's tau      : {res.avg_kendalls_

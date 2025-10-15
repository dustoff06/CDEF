# src/cdef_analyzer/demo.py
from __future__ import annotations

import argparse
import sys
import csv
from typing import Optional

from .gumbel_copula_fixed import RankDependencyAnalyzer, CopulaResults


def format_summary(res: CopulaResults) -> str:
    lines = []
    lines.append("=" * 76)
    lines.append("CDEF / Gumbel Copula — Summary")
    lines.append("=" * 76)
    lines.append(f"Items × Raters            : {res.n_items} × {res.n_raters}")
    lines.append(f"Ranking type              : {res.ranking_type}")
    lines.append(f"Distribution model        : {res.distribution_model}")
    if res.model_log_likelihood is not None:
        lines.append(f"Model log-likelihood      : {res.model_log_likelihood}")
    lines.append("")
    lines.append("Core metrics")
    lines.append(f"  Kendall's W (concord.)  : {res.kendalls_W:.3f}")
    lines.append(f"  Avg Kendall's tau       : {res.avg_kendalls_tau:.3f}")
    lines.append(f"  θ (scaled with W)       : {res.theta_scaled:.3f}")
    lines.append(f"  θ (from τ)              : {res.theta_gumbel:.3f}")
    lines.append(f"  Mutual information      : {res.mutual_information:.3f}")
    lines.append("")
    lines.append("Dependence test (χ² on binned pairs)")
    lines.append(f"  χ² statistic            : {res.chi_square_stat:.3f}")
    lines.append(f"  p-value                 : {res.p_value:.3f}")
    lines.append("")
    lines.append("Log-likelihoods")
    lines.append(f"  Copula (average)        : {res.avg_log_likelihood:.6f}")
    lines.append(f"  Independence baseline   : {res.independence_log_likelihood:.6f}")
    lines.append("")
    lines.append("Relative importance (weights, not probabilities)")
    for k, v in res.relative_importance.items():
        lines.append(f"  {k:15s}: {v:.3f}")
    lines.append("")
    lines.append("Pairwise θ (Gumbel) — first few")
    shown = 0
    for pair, theta in res.pairwise_thetas.items():
        lines.append(f"  {pair:>15s}: {theta:.3f}")
        shown += 1
        if shown >= 8:
            if len(res.pairwise_thetas) > shown:
                lines.append("  …")
            break
    lines.append("=" * 76)
    return "\n".join(lines)


def write_csv(res: CopulaResults, path: str) -> None:
    """Write a one-row CSV of headline metrics."""
    fieldnames = [
        "n_items",
        "n_raters",
        "ranking_type",
        "distribution_model",
        "model_log_likelihood",
        "kendalls_W",
        "avg_kendalls_tau",
        "theta_scaled",
        "theta_gumbel",
        "mutual_information",
        "chi_square_stat",
        "p_value",
        "avg_log_likelihood",
        "independence_log_likelihood",
        "rel_concordance",
        "rel_concurrence",
        "rel_extremeness",
    ]
    row = {
        "n_items": res.n_items,
        "n_raters": res.n_raters,
        "ranking_type": res.ranking_type,
        "distribution_model": res.distribution_model,
        "model_log_likelihood": res.model_log_likelihood,
        "kendalls_W": f"{res.kendalls_W:.6f}",
        "avg_kendalls_tau": f"{res.avg_kendalls_tau:.6f}",
        "theta_scaled": f"{res.theta_scaled:.6f}",
        "theta_gumbel": f"{res.theta_gumbel:.6f}",
        "mutual_information": f"{res.mutual_information:.6f}",
        "chi_square_stat": f"{res.chi_square_stat:.6f}",
        "p_value": f"{res.p_value:.6f}",
        "avg_log_likelihood": f"{res.avg_log_likelihood:.6f}",
        "independence_log_likelihood": f"{res.independence_log_likelihood:.6f}",
        "rel_concordance": f"{res.relative_importance.get('Concordance', 0.0):.6f}",
        "rel_concurrence": f"{res.relative_importance.get('Concurrence', 0.0):.6f}",
        "rel_extremeness": f"{res.relative_importance.get('Extremeness', 0.0):.6f}",
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m cdef_analyzer.demo",
        description="Run CDEF / Gumbel-copula analysis from an Excel file in long format.",
    )
    p.add_argument("--excel", required=True, help="Path to Excel file (long format).")
    p.add_argument("--sheet", default="Sheet1", help="Sheet name (default: Sheet1).")
    p.add_argument("--rater-col", required=True, help="Column name for raters.")
    p.add_argument("--ratee-col", required=True, help="Column name for items (ratees).")
    p.add_argument("--ranking-col", required=True, help="Column name for rank values.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--out-csv", default=None, help="Optional path to write a summary CSV.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    analyzer = RankDependencyAnalyzer(random_seed=args.seed)

    try:
        res = analyzer.analyze_from_excel(
            file_path=args.excel,
            sheet_name=args.sheet,
            rater_col=args.rater_col,
            ratee_col=args.ratee_col,
            ranking_col=args.ranking_col,
        )
    except Exception as e:
        # Provide a clear message and a non-zero exit code.
        sys.stderr.write(f"[cdef_analyzer.demo] Error: {e}\n")
        return 2

    print(format_summary(res))

    if args.out_csv:
        try:
            write_csv(res, args.out_csv)
            print(f"\nWrote summary CSV → {args.out_csv}")
        except Exception as e:
            sys.stderr.write(f"[cdef_analyzer.demo] Failed to write CSV: {e}\n")
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from cdef_analyzer.demo import main

if __name__ == "__main__":
    # Run the exemplar with the repo’s example data
    raise SystemExit(main([
        "--excel", "src/examples/data.xlsx",
        "--sheet", "Sheet1",
        "--rater-col", "Rater",
        "--ratee-col", "Ratee",
        "--ranking-col", "Ranking",
        "--seed", "42",
    ]))



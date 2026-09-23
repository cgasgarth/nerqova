"""Check matched Kev MLX and Nerqova benchmark or evaluation reports.

    uv run python scripts/compare_engines.py runs/kev-mlx-latency.json runs/nerqova-latency.json
    uv run python scripts/compare_engines.py --evaluation runs/kev-mlx-development runs/nerqova-development
"""

import argparse
import json
from pathlib import Path

from nerqova.comparison import compare_benchmarks, compare_evaluations, compare_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="store_true", help="arguments are evaluation directories")
    parser.add_argument("--complete", action="store_true", help="arguments are complete-decision latency reports")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    if args.evaluation and args.complete:
        parser.error("choose one report type")
    if args.evaluation:
        def load(directory):
            directory = Path(directory)
            report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            rows = {part: json.loads((directory / part / "rows.json").read_text(encoding="utf-8"))
                    for part in ("development", "transfer")}
            return report, rows

        reference, reference_rows = load(args.reference)
        candidate, candidate_rows = load(args.candidate)
        result = compare_evaluations(reference, candidate, reference_rows, candidate_rows)
    else:
        with open(args.reference, encoding="utf-8") as stream:
            reference = json.load(stream)
        with open(args.candidate, encoding="utf-8") as stream:
            candidate = json.load(stream)
        result = compare_latency(reference, candidate) if args.complete else compare_benchmarks(reference, candidate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Check matched Kev MLX and Nerqova benchmark reports.

    uv run python scripts/compare_engines.py runs/kev-mlx-latency.json runs/nerqova-latency.json
"""

import argparse
import json

from nerqova.comparison import compare_benchmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    with open(args.reference, encoding="utf-8") as stream:
        reference = json.load(stream)
    with open(args.candidate, encoding="utf-8") as stream:
        candidate = json.load(stream)
    print(json.dumps(compare_benchmarks(reference, candidate), indent=2))


if __name__ == "__main__":
    main()

"""Screen stricter exit gaps from one development-only mixed prediction run.

The input early run must use a gap no larger than every screened gap. Rows that
already fell back remain full-model rows. Re-run the selected gap through the
model before claiming its quality or speed.
"""

import argparse
import json
from pathlib import Path

from kev.metrics import metrics

from select_exit_gate import top_two_gap


def aligned_rows(reference, candidate, part):
    def read(directory):
        return json.loads((directory / part / "rows.json").read_text())

    reference_rows, candidate_rows = read(reference), read(candidate)
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(f"{part} row counts differ")
    for full, early in zip(reference_rows, candidate_rows, strict=True):
        identity = ("id", "question", "variant", "type", "label", "keys")
        if any(full[key] != early[key] for key in identity):
            raise ValueError(f"{part} rows are not aligned")
    return reference_rows, candidate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--gaps", type=float, nargs="+", default=[2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    full_report = json.loads((args.reference / "result.json").read_text())
    early_report = json.loads((args.candidate / "result.json").read_text())
    if (full_report["run"] != early_report["run"] or
            full_report["checkpoint"] != early_report["checkpoint"] or
            full_report["served_temperature"] != early_report["served_temperature"] or
            full_report["locked_test_read"] or early_report["locked_test_read"]):
        raise ValueError("runs differ or include locked test data")
    if any(gap < early_report["exit_gap"] for gap in args.gaps):
        raise ValueError("screened gaps must be at least the candidate run's gap")

    report = {"run": full_report["run"], "candidate_gap": early_report["exit_gap"], "screen": {}}
    for part in ("development", "transfer"):
        full_rows, early_rows = aligned_rows(args.reference, args.candidate, part)
        entries = []
        for gap in args.gaps:
            chosen = []
            flips = 0
            for full, early in zip(full_rows, early_rows, strict=True):
                use_early = top_two_gap(early["p"]) >= gap
                row = early if use_early else full
                chosen.append(row)
                flips += int(max(range(len(row["p"])), key=row["p"].__getitem__) !=
                             max(range(len(full["p"])), key=full["p"].__getitem__))
            clean = [row for row in chosen if row["variant"] == "clean"]
            score = metrics(clean)
            entries.append({"gap": gap, "choice_flips": flips,
                            "accuracy": score["acc"], "brier": score["brier"], "ece": score["ece"]})
        report["screen"][part] = entries
    output = json.dumps(report, indent=2) + "\n"
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)


if __name__ == "__main__":
    main()

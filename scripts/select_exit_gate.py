"""Measure confidence gates on the held-out calibration partition.

The saved teacher distribution is the exact packed full-model fallback. This
script does not read development or locked test data.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev.metrics import metrics
from kev.suite import digest


def quality(probabilities, labels):
    return metrics([{"p": p, "label": int(y), "type": "choice"}
                    for p, y in zip(probabilities, labels, strict=True)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--verify-head", type=Path)
    parser.add_argument("--verify-threshold", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (args.verify_head is None) != (args.verify_threshold is None):
        parser.error("verify-head and verify-threshold must be set together")
    with np.load(args.artifact.with_suffix(".calibration.npz")) as data:
        offsets = data["offsets"]
        students = [data["student"][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        teachers = [data["teacher"][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        labels = data["labels"]
        records = data["record_numbers"]
    verifier = None
    if args.verify_head:
        with np.load(args.verify_head.with_suffix(".calibration.npz")) as data:
            if (not np.array_equal(data["offsets"], offsets)
                    or not np.array_equal(data["labels"], labels)
                    or not np.array_equal(data["record_numbers"], records)):
                raise ValueError("verifier calibration rows differ from exit-head rows")
            verifier = [data["student"][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    baseline = quality(teachers, labels)
    student = quality(students, labels)
    thresholds = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]
    sweep = []
    for threshold in thresholds:
        accepted = np.asarray([
            float(max(p)) >= threshold and
            (verifier is None or (
                (float(max(verifier[index])) - 1 / len(verifier[index]))
                / (1 - 1 / len(verifier[index])) >= args.verify_threshold
                and int(np.argmax(verifier[index])) == int(np.argmax(p))))
            for index, p in enumerate(students)
        ])
        mixed = [s if use else t for s, t, use in zip(students, teachers, accepted)]
        score = quality(mixed, labels)
        record_exit = defaultdict(list)
        for record, use in zip(records, accepted):
            record_exit[int(record)].append(bool(use))
        teacher_flips = sum(bool(np.argmax(s) != np.argmax(t)) and use
                            for s, t, use in zip(students, teachers, accepted))
        sweep.append({"threshold": threshold,
                      "question_exit_rate": float(accepted.mean()),
                      "complete_record_exit_rate": float(np.mean([all(v) for v in record_exit.values()])),
                      "teacher_choice_flips": int(teacher_flips),
                      "accuracy": score["acc"], "brier": score["brier"], "ece": score["ece"],
                      "nll": score["nll"]})
    report = {"questions": len(labels), "records": len(set(records)),
              "exit_head_sha256": digest(args.artifact),
              "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
              "verify_threshold": args.verify_threshold,
              "full": {k: baseline[k] for k in ("acc", "brier", "ece", "nll")},
              "early_only": {k: student[k] for k in ("acc", "brier", "ece", "nll")},
              "gates": sweep}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

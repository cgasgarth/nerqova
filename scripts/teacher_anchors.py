"""Turn a Kev training-partition benchmark into keyed teacher targets.

    uv run python scripts/teacher_anchors.py \
      --teacher /tmp/kev4b-v7-train-teacher \
      --suite evals/v7/decision-v7 \
      --out runs/student-teacher-anchors.json

The output works with `kev.train --anchor ... --anchor_w ...`. Targets use
the teacher's served probabilities; the trainer skips altered option sets.
"""

import argparse
import math
from pathlib import Path

from kev.suite import digest, read_json, write_json


def build(teacher_dir, suite, out):
    teacher_dir, suite, out = Path(teacher_dir), Path(suite), Path(out)
    report = read_json(teacher_dir / "report.json")
    suite_hash = digest(suite / "manifest.json")
    if report.get("split") != "train" or report.get("suite_sha256") != suite_hash:
        raise ValueError("teacher rows must come from this suite's training partition")
    coverage = report["coverage"]
    if coverage["rejected_records"] or coverage["truncated_records"] or coverage["evaluated_questions"] != coverage["requested_questions"]:
        raise ValueError("teacher benchmark must cover every training question without rejection or truncation")

    rows_path = teacher_dir / "rows.json"
    rows = read_json(rows_path)
    if len(rows) != coverage["requested_questions"]:
        raise ValueError("teacher row count does not match the benchmark report")
    targets = {}
    for row in rows:
        if row["variant"] != "clean":
            raise ValueError("teacher rows must be canonical training questions")
        keys, values = row["keys"], row["p"]
        if len(keys) != len(values) or len(set(keys)) != len(keys) or not all(math.isfinite(p) and 0 <= p <= 1 for p in values) or abs(sum(values) - 1) > 1e-5:
            raise ValueError(f"invalid teacher distribution for {row['id']}:{row['question']}")
        questions = targets.setdefault(row["id"], {})
        if row["question"] in questions:
            raise ValueError(f"duplicate teacher question {row['id']}:{row['question']}")
        questions[row["question"]] = dict(zip(keys, values))

    output = {"_meta": {"teacher": report["run"], "teacher_temperature": report["calibration"]["inference_temperature"],
                        "suite": str(suite), "suite_sha256": suite_hash, "partition": "train", "rows_sha256": digest(rows_path),
                        "records": len(targets), "questions": len(rows), "readout": "served Kev probability vectors"},
              "targets": targets}
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, output)
    return output["_meta"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.teacher, args.suite, args.out))


if __name__ == "__main__":
    main()

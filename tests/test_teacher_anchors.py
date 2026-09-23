"""Teacher targets must cover the frozen training partition only."""

import pytest

from kev.suite import digest, read_json, write_json
from scripts.teacher_anchors import build


def teacher_fixture(tmp_path):
    suite, teacher = tmp_path / "suite", tmp_path / "teacher"
    suite.mkdir()
    teacher.mkdir()
    write_json(suite / "manifest.json", {"version": 1})
    report = {
        "run": "pinned-teacher",
        "split": "train",
        "suite_sha256": digest(suite / "manifest.json"),
        "calibration": {"inference_temperature": 2.14},
        "coverage": {
            "requested_questions": 1,
            "evaluated_questions": 1,
            "rejected_records": 0,
            "truncated_records": 0,
        },
    }
    write_json(teacher / "report.json", report)
    write_json(
        teacher / "rows.json",
        [{"id": "item-0", "question": "reason", "variant": "clean",
          "keys": ["size", "damage"], "p": [0.8, 0.2]}],
    )
    return suite, teacher, report


def test_teacher_targets_preserve_option_keys_and_provenance(tmp_path):
    suite, teacher, _ = teacher_fixture(tmp_path)
    destination = tmp_path / "nested" / "anchors.json"
    build(teacher, suite, destination)
    anchors = read_json(destination)
    assert anchors["targets"] == {"item-0": {"reason": {"size": 0.8, "damage": 0.2}}}
    assert anchors["_meta"]["partition"] == "train"
    assert anchors["_meta"]["rows_sha256"] == digest(teacher / "rows.json")


@pytest.mark.parametrize("change", [
    {"split": "development"},
    {"suite_sha256": "wrong-suite"},
    {"coverage": {"requested_questions": 2, "evaluated_questions": 1,
                  "rejected_records": 0, "truncated_records": 0}},
])
def test_teacher_targets_reject_wrong_partition_or_incomplete_run(tmp_path, change):
    suite, teacher, report = teacher_fixture(tmp_path)
    write_json(teacher / "report.json", {**report, **change})
    with pytest.raises(ValueError):
        build(teacher, suite, tmp_path / "anchors.json")


def test_teacher_targets_reject_duplicate_question(tmp_path):
    suite, teacher, report = teacher_fixture(tmp_path)
    row = read_json(teacher / "rows.json")[0]
    write_json(teacher / "rows.json", [row, row])
    report["coverage"]["requested_questions"] = 2
    report["coverage"]["evaluated_questions"] = 2
    write_json(teacher / "report.json", report)
    with pytest.raises(ValueError, match="duplicate"):
        build(teacher, suite, tmp_path / "anchors.json")

import pytest

from nerqova.comparison import compare_benchmarks, compare_rows


def report(engine, p):
    return {
        "engine": engine, "checkpoint_revision": "revision", "base_revision": "base",
        "adapter_sha256": "adapter", "head_sha256": "head", "temperature": 2.1,
        "encoded_request_sha256": "request", "fixture_probabilities": [p],
        "latency_ms": {"new_state": {"median": 200 if engine == "kev-mlx" else 100},
                       "cached_state": {"median": 80 if engine == "kev-mlx" else 40}},
    }


def test_comparison_requires_identical_checkpoint_and_request():
    reference = report("kev-mlx", [0.8, 0.2])
    candidate = report("nerqova", [0.799, 0.201])
    result = compare_benchmarks(reference, candidate)
    assert result["choice_flips"] == 0
    assert result["new_state_speedup"] == result["cached_state_speedup"] == 2
    candidate["head_sha256"] = "other"
    with pytest.raises(ValueError, match="head_sha256"):
        compare_benchmarks(reference, candidate)


def test_row_comparison_aligns_question_ids_and_catches_choice_flip():
    def row(qid, p):
        return {"id": "item", "question": qid, "keys": ["yes", "no"], "label": 0,
                "source": "case", "variant": "clean", "p": p}

    baseline = [row("first", [0.9, 0.1]), row("second", [0.51, 0.49])]
    changed = [row("second", [0.49, 0.51]), row("first", [0.89, 0.11])]
    result = compare_rows(baseline, changed)
    assert result["questions"] == 2
    assert result["choice_flips"] == 1
    assert result["max_probability_delta"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="rows differ"):
        compare_rows(baseline, changed[:1])

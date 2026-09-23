"""Compare two engines on identical requests and checkpoint weights."""


IDENTITY = ("checkpoint_revision", "base_revision", "adapter_sha256", "head_sha256",
            "temperature", "encoded_request_sha256")


def compare_benchmarks(reference, candidate):
    for field in IDENTITY:
        if reference[field] != candidate[field]:
            raise ValueError(f"benchmark {field} differs")
    baseline = reference["fixture_probabilities"]
    changed = candidate["fixture_probabilities"]
    if len(baseline) != len(changed) or any(len(a) != len(b) for a, b in zip(baseline, changed)):
        raise ValueError("benchmark question or option counts differ")
    deltas = [abs(a - b) for first, second in zip(baseline, changed)
              for a, b in zip(first, second)]
    flips = sum(max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__)
                for a, b in zip(baseline, changed))
    return {
        "questions": len(baseline),
        "max_probability_delta": max(deltas),
        "choice_flips": flips,
        "new_state_speedup": reference["latency_ms"]["new_state"]["median"] / candidate["latency_ms"]["new_state"]["median"],
        "cached_state_speedup": reference["latency_ms"]["cached_state"]["median"] / candidate["latency_ms"]["cached_state"]["median"],
    }


def compare_rows(reference, candidate):
    def indexed(rows):
        result = {(row["id"], row["question"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate benchmark row")
        return result

    baseline, changed = indexed(reference), indexed(candidate)
    if set(baseline) != set(changed):
        raise ValueError("benchmark rows differ")
    max_delta = 0.0
    flips = 0
    for key in baseline:
        first, second = baseline[key], changed[key]
        if any(first[field] != second[field] for field in ("keys", "label", "source", "variant")):
            raise ValueError(f"benchmark labels or option order differ: {key}")
        if len(first["p"]) != len(second["p"]):
            raise ValueError(f"benchmark option count differs: {key}")
        max_delta = max(max_delta, *(abs(a - b) for a, b in zip(first["p"], second["p"])))
        flips += max(range(len(first["p"])), key=first["p"].__getitem__) != max(
            range(len(second["p"])), key=second["p"].__getitem__)
    return {"questions": len(baseline), "max_probability_delta": max_delta, "choice_flips": flips}

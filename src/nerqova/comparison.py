"""Compare two engines on identical requests and checkpoint weights."""


IDENTITY = ("checkpoint_revision", "base_revision", "adapter_sha256", "head_sha256",
            "temperature", "encoded_request_sha256", "code_sha")


def compare_latency(reference, candidate):
    for field in IDENTITY:
        if reference[field] != candidate[field]:
            raise ValueError(f"benchmark {field} differs")
    if "requests_sha256" in reference or "requests_sha256" in candidate:
        if reference.get("requests_sha256") != candidate.get("requests_sha256"):
            raise ValueError("benchmark HTTP requests differ")
    if "choices" in reference or "choices" in candidate:
        if reference.get("choices") != candidate.get("choices"):
            raise ValueError("benchmark choices differ")
    if "input_tokens" in reference or "input_tokens" in candidate:
        if reference.get("input_tokens") != candidate.get("input_tokens"):
            raise ValueError("benchmark input-token counts differ")
    return {
        "new_state_speedup": reference["latency_ms"]["new_state"]["median"] / candidate["latency_ms"]["new_state"]["median"],
        "cached_state_speedup": reference["latency_ms"]["cached_state"]["median"] / candidate["latency_ms"]["cached_state"]["median"],
    }


def compare_benchmarks(reference, candidate):
    speed = compare_latency(reference, candidate)
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
        **speed,
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


def compare_evaluations(reference, candidate, reference_rows, candidate_rows):
    identity = ("checkpoint", "base_revision", "adapter_sha256", "head_sha256",
                "served_temperature", "suite_sha256", "transfer_suite_sha256")
    for field in identity:
        if reference[field] != candidate[field]:
            raise ValueError(f"evaluation {field} differs")
    if set(reference_rows) != {"development", "transfer"} or set(candidate_rows) != set(reference_rows):
        raise ValueError("development and transfer rows are required")
    return {part: compare_rows(reference_rows[part], candidate_rows[part])
            for part in ("development", "transfer")}

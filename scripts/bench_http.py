"""Time complete HTTP decisions against one resident local server.

Run stock and Nerqova servers in separate quiet windows, then compare reports
with scripts/compare_engines.py --complete. Unique states measure cache misses;
one repeated state measures cache hits.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint
from kev.model import encode, load_tokenizer
from kev.suite import digest

from nerqova.workload import workload


def measure(client, payloads):
    samples = []
    last = None
    for payload in payloads:
        started = time.perf_counter()
        response = client.post("/v1/systemone", json=payload)
        response.raise_for_status()
        last = response.json()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {"median": round(statistics.median(samples), 2),
            "p95": round(samples[int((len(samples) - 1) * 0.95)], 2),
            "samples": [round(value, 2) for value in samples]}, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["kev-mlx", "nerqova-packed", "nerqova-early"], required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--run", default="jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778")
    parser.add_argument("--exit-head", type=Path)
    parser.add_argument("--exit-gap", type=float)
    parser.add_argument("--exit-gap-wide", type=float)
    parser.add_argument("--verify-head", type=Path)
    parser.add_argument("--verify-threshold", type=float)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--choices", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.engine == "nerqova-early" and (args.exit_head is None or args.exit_gap is None):
        parser.error("nerqova-early requires --exit-head and --exit-gap for provenance")
    if args.engine != "nerqova-early" and (args.exit_head is not None or args.exit_gap is not None or args.exit_gap_wide is not None):
        parser.error("exit options require nerqova-early")
    if args.exit_gap_wide is not None and args.exit_gap is None:
        parser.error("exit-gap-wide requires exit-gap")
    if (args.verify_head is None) != (args.verify_threshold is None) or (args.verify_head and args.engine != "nerqova-early"):
        parser.error("verifier options must be set together for nerqova-early")
    if args.reps < 1 or args.warmups < 0 or args.questions < 1 or args.choices < 2:
        parser.error("reps and questions must be positive; warmups nonnegative; choices at least two")

    checkpoint = Checkpoint(args.run)
    tok = load_tokenizer(checkpoint.meta.base, revision=checkpoint.meta.base_revision)
    example = workload(tok, 270, args.questions, args.choices)
    request = {
        "state": example["state"], "model": "kev-latest",
        "questions": {str(i): {"type": "choice", "instructions": q["instr"],
                               "criteria": {option: None for option in q["options"]}}
                      for i, q in enumerate(example["questions"])},
    }
    record, _ = to_record(SystemOneRequest.model_validate(request))
    encoded = encode(tok, record)
    warm_new = [{**request, "state": request["state"] + f" Ticket {i:05d}."}
                for i in range(args.warmups)]
    measured_new = [{**request, "state": request["state"] + f" Ticket {i + args.warmups:05d}."}
                    for i in range(args.reps)]
    request_hash = hashlib.sha256(json.dumps([warm_new, measured_new, request], sort_keys=True).encode()).hexdigest()

    with httpx.Client(base_url=args.url, timeout=120) as client:
        runtime = None
        if args.engine != "kev-mlx":
            response = client.get("/v1/nerqova")
            response.raise_for_status()
            runtime = response.json()
            expected = {"engine": args.engine,
                        "checkpoint_revision": Path(checkpoint.path).name,
                        "exit_head_sha256": digest(args.exit_head) if args.exit_head else None,
                        "exit_gap": args.exit_gap,
                        "exit_gap_wide": args.exit_gap_wide,
                        "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
                        "verify_threshold": args.verify_threshold}
            if runtime != expected:
                raise ValueError(f"server runtime does not match requested benchmark: {runtime}")
        for payload in warm_new:
            response = client.post("/v1/systemone", json=payload)
            response.raise_for_status()
        fresh, _ = measure(client, measured_new)
        for _ in range(args.warmups):
            response = client.post("/v1/systemone", json=request)
            response.raise_for_status()
        cached, last = measure(client, [request] * args.reps)

    report = {
        "engine": args.engine,
        "server_runtime": runtime,
        "transport": "HTTP loopback",
        "url": args.url,
        "run": args.run,
        "checkpoint_revision": Path(checkpoint.path).name,
        "base_revision": checkpoint.meta.base_revision,
        "adapter_sha256": digest(checkpoint.file("adapter_model.safetensors")),
        "head_sha256": digest(checkpoint.file("head.pt")),
        "temperature": checkpoint.meta.temperature,
        "exit_head_sha256": digest(args.exit_head) if args.exit_head else None,
        "exit_gap": args.exit_gap,
        "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
        "verify_threshold": args.verify_threshold,
        "encoded_request_sha256": hashlib.sha256(json.dumps(encoded["ids"]).encode()).hexdigest(),
        "requests_sha256": request_hash,
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "warmups": args.warmups,
        "reps": args.reps,
        "questions": args.questions,
        "choices_per_question": args.choices,
        "input_tokens": last["usage"]["input_tokens"],
        "choices": {key: answer["choice"] for key, answer in last["answers"].items()},
        "latency_ms": {"new_state": fresh, "cached_state": cached},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"engine": args.engine, "latency_ms": {
        key: {field: values[field] for field in ("median", "p95")}
        for key, values in report["latency_ms"].items()}}, indent=2))


if __name__ == "__main__":
    main()

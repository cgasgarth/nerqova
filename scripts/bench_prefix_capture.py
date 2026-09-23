"""Compare complete single-question decisions with and without a prefix snapshot.

    uv run python scripts/bench_prefix_capture.py --out runs/prefix-capture.json
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from kev.api import SystemOneRequest, to_record
from kev.serve import Server
from kev.suite import digest

from nerqova.checkpoint import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="jaredpalmer/kev-4b")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.warmups < 0:
        parser.error("reps must be positive and warmups must be nonnegative")
    checkpoint, tok, model = load_model(args.run, packed_delta=True)
    loaded = {"active_mib": mx.get_active_memory() / 2**20,
              "cache_mib": mx.get_cache_memory() / 2**20,
              "peak_mib": mx.get_peak_memory() / 2**20,
              "rss_mib": int(subprocess.check_output(
                  ["ps", "-p", str(os.getpid()), "-o", "rss="], text=True)) / 1024}
    request = SystemOneRequest.model_validate({
        "state": "2048 board, rows top to bottom, 0 means empty:\n0 0 0 0\n0 0 0 0\n2 0 0 0\n0 0 0 2\nScore: 0.",
        "questions": {"move": {"type": "choice",
            "instructions": "Choose the next move to reach 2048. Merge tiles and keep open cells. Only legal moves are listed.",
            "criteria": {d: f"Slide tiles {d}." for d in ["up", "right", "down", "left"]}}},
    })
    encoded = model.encode(tok, to_record(request)[0])
    reference = model._branch_probs(encoded, model.prefix(encoded)[1])
    actual, prefix = model.probs_and_prefix(encoded)
    hit = model.probs_with_prefix(encoded, prefix)
    probability_delta = max(float((a - b).abs().max()) for group in (actual, hit)
                            for a, b in zip(reference, group))
    server = Server(checkpoint, tok, model, "mps")
    measurements = []
    for capture in (False, True, False, True):
        # DeltaNet remains packed in both modes; only the request path changes.
        model.packed_delta = capture
        timings = {}
        for cold in (True, False):
            values = []
            for index in range(args.warmups + args.reps):
                if cold:
                    server.prefix_cache.clear()
                start = time.perf_counter()
                response = server.answer(request)
                elapsed = (time.perf_counter() - start) * 1000
                if index >= args.warmups:
                    values.append(elapsed)
            timings["cold" if cold else "cached"] = {
                "median_ms": statistics.median(values),
                "p95_ms": sorted(values)[int((len(values) - 1) * 0.95)],
                "samples_ms": values, "answers": response["answers"],
            }
        measurements.append({"capture_prefix": capture, "timings": timings})
    result = {"checkpoint_revision": Path(checkpoint.path).name,
              "base_revision": checkpoint.meta.base_revision,
              "adapter_sha256": digest(checkpoint.file("adapter_model.safetensors")),
              "head_sha256": digest(checkpoint.file("head.pt")),
              "temperature": model.head.temperature,
              "encoded_request_sha256": hashlib.sha256(json.dumps(encoded["ids"]).encode()).hexdigest(),
              "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
              "loaded_memory": loaded, "max_probability_delta": probability_delta,
              "all_choices_equal": all(a.argmax().item() == b.argmax().item()
                                       for group in (actual, hit) for a, b in zip(reference, group)),
              "reps": args.reps, "warmups": args.warmups, "measurements": measurements}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "max_probability_delta": probability_delta,
                      "median_ms": [{"capture": x["capture_prefix"], **{
                          k: round(v["median_ms"], 2) for k, v in x["timings"].items()
                      }} for x in measurements]}))


if __name__ == "__main__":
    main()

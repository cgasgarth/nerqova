"""Compare stock Kev MLX and Nerqova model-only decisions on the local GPU.

    uv run python scripts/bench_decisions.py --engine kev-mlx --run jaredpalmer/kev-4b
    uv run python scripts/bench_decisions.py --engine nerqova --run jaredpalmer/kev-4b

The fixed workload has a roughly 270-token state and five three-option questions.
Both paths take a pre-encoded request and return host-visible probabilities, so
timings include GPU completion but exclude encoding, formatting and HTTP.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from kev.checkpoint import Checkpoint, LoadOptions
from kev.suite import digest
from nerqova.checkpoint import load_model
from nerqova.workload import workload


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def measure(fn, reps, warmups=10):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "median": round(statistics.median(samples), 2),
        "p10": round(samples[int((reps - 1) * 0.1)], 2),
        "p90": round(samples[int((reps - 1) * 0.9)], 2),
        "p95": round(samples[int((reps - 1) * 0.95)], 2),
        "samples": [round(value, 2) for value in samples],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="jaredpalmer/kev-4b")
    parser.add_argument("--engine", choices=["kev-mlx", "nerqova", "nerqova-unmasked", "nerqova-packed", "nerqova-early"], default="nerqova")
    parser.add_argument("--exit-head", type=Path)
    parser.add_argument("--exit-threshold", type=float)
    parser.add_argument("--verify-head", type=Path)
    parser.add_argument("--verify-threshold", type=float)
    parser.add_argument("--state-tokens", type=int, default=270)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.engine == "nerqova-early" and (args.exit_head is None or args.exit_threshold is None):
        parser.error("nerqova-early requires --exit-head and --exit-threshold")
    if args.engine != "nerqova-early" and (args.exit_head is not None or args.exit_threshold is not None):
        parser.error("exit options require nerqova-early")
    if (args.verify_head is None) != (args.verify_threshold is None) or (args.verify_head and args.engine != "nerqova-early"):
        parser.error("verifier options must be set together for nerqova-early")
    if args.state_tokens < 1 or args.questions < 1 or args.reps < 1 or args.warmups < 0:
        parser.error("state-tokens, questions and reps must be positive; warmups must be nonnegative")

    if args.engine == "kev-mlx":
        checkpoint = Checkpoint(args.run)
        tok, model = checkpoint.load("mps", LoadOptions(backend="mlx"))
    else:
        checkpoint, tok, model = load_model(
            args.run, unmasked_branches=args.engine == "nerqova-unmasked",
            packed_delta=args.engine in ("nerqova-packed", "nerqova-early"),
            exit_head=args.exit_head, exit_threshold=args.exit_threshold,
            verify_head=args.verify_head, verify_threshold=args.verify_threshold,
        )
    enc = model.encode(tok, workload(tok, args.state_tokens, args.questions))
    fixture_probabilities, prefix = model.probs_and_prefix(enc)
    result = {
        "engine": args.engine,
        "run": args.run,
        "checkpoint_revision": Path(checkpoint.path).name,
        "base": checkpoint.meta.base,
        "base_revision": checkpoint.meta.base_revision,
        "adapter_sha256": digest(checkpoint.file("adapter_model.safetensors")),
        "head_sha256": digest(checkpoint.file("head.pt")),
        "temperature": model.head.temperature,
        "exit_head_sha256": digest(args.exit_head) if args.exit_head else None,
        "exit_threshold": args.exit_threshold,
        "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
        "verify_threshold": args.verify_threshold,
        "code_sha": command("git", "rev-parse", "HEAD"),
        "code_dirty": bool(command("git", "status", "--porcelain")),
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "dtype": model.dtype,
        "state_tokens": enc["seg"].count(0),
        "packed_tokens": len(enc["ids"]),
        "questions": len(enc["decide_idx"]),
        "warmups": args.warmups,
        "encoded_request_sha256": hashlib.sha256(json.dumps(enc["ids"]).encode()).hexdigest(),
        "fixture_probabilities": [p.tolist() for p in fixture_probabilities],
        "latency_ms": {
            "new_state": measure(lambda: model.probs(enc), args.reps, args.warmups),
            "cached_state": measure(lambda: model.probs_with_prefix(enc, prefix), args.reps, args.warmups),
        },
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

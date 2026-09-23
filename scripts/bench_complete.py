"""Time a complete in-process System One decision, including encoding and formatting.

    uv run python scripts/bench_complete.py --engine kev-mlx --out runs/kev-mlx-complete.json
    uv run python scripts/bench_complete.py --engine nerqova --out runs/nerqova-complete.json

Each engine runs in a separate process. HTTP transport is measured separately.
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

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.serve import Server
from kev.suite import digest

from nerqova.checkpoint import load_model
from nerqova.workload import workload


def measure(server, request, reps, warmups, new_state):
    def once():
        if new_state:
            server.prefix_cache.clear()
        start = time.perf_counter()
        response = server.answer(request)
        return (time.perf_counter() - start) * 1000, response

    for _ in range(warmups):
        once()
    samples = [once()[0] for _ in range(reps)]
    samples.sort()
    return {"median": round(statistics.median(samples), 2),
            "p95": round(samples[int((reps - 1) * 0.95)], 2),
            "samples": [round(value, 2) for value in samples]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["kev-mlx", "nerqova"], required=True)
    parser.add_argument("--run", default="jaredpalmer/kev-4b")
    parser.add_argument("--state-tokens", type=int, default=270)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.reps < 1 or args.warmups < 0 or args.state_tokens < 1 or args.questions < 1:
        parser.error("reps, state-tokens and questions must be positive; warmups must be nonnegative")

    if args.engine == "kev-mlx":
        checkpoint = Checkpoint(args.run)
        tok, model = checkpoint.load("mps", LoadOptions(backend="mlx"))
    else:
        checkpoint, tok, model = load_model(args.run)
    example = workload(tok, args.state_tokens, args.questions)
    request = SystemOneRequest.model_validate({
        "state": example["state"],
        "questions": {str(i): {"type": "choice", "instructions": q["instr"],
                               "criteria": {option: None for option in q["options"]}}
                      for i, q in enumerate(example["questions"])},
    })
    encoded = model.encode(tok, to_record(request)[0])
    server = Server(checkpoint, tok, model, "mps")
    response = server.answer(request)
    result = {
        "engine": args.engine,
        "run": args.run,
        "checkpoint_revision": Path(checkpoint.path).name,
        "base_revision": checkpoint.meta.base_revision,
        "adapter_sha256": digest(checkpoint.file("adapter_model.safetensors")),
        "head_sha256": digest(checkpoint.file("head.pt")),
        "temperature": model.head.temperature,
        "encoded_request_sha256": hashlib.sha256(json.dumps(encoded["ids"]).encode()).hexdigest(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "state_tokens": encoded["seg"].count(0),
        "packed_tokens": len(encoded["ids"]),
        "questions": len(example["questions"]),
        "choices": {key: answer["choice"] for key, answer in response["answers"].items()},
        "warmups": args.warmups,
        "latency_ms": {
            "new_state": measure(server, request, args.reps, args.warmups, True),
            "cached_state": measure(server, request, args.reps, args.warmups, False),
        },
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

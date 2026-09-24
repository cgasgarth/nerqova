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
import resource
import re
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

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

    mx.reset_peak_memory()
    for _ in range(warmups):
        once()
    samples = [once()[0] for _ in range(reps)]
    samples.sort()
    median = statistics.median(samples)
    return {"median": round(median, 2),
            "serial_requests_per_second": round(1000 / median, 2),
            "p95": round(samples[int((reps - 1) * 0.95)], 2),
            "samples": [round(value, 2) for value in samples],
            "memory_gib": {
                "mlx_active": round(mx.get_active_memory() / 1024 ** 3, 3),
                "mlx_peak": round(mx.get_peak_memory() / 1024 ** 3, 3),
                "mlx_reusable_cache": round(mx.get_cache_memory() / 1024 ** 3, 3),
                "process_peak_rss": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3, 3),
            }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["kev-mlx", "nerqova", "nerqova-unmasked", "nerqova-packed", "nerqova-early"], required=True)
    parser.add_argument("--exit-head", type=Path)
    parser.add_argument("--exit-gap", type=float)
    parser.add_argument("--exit-gap-wide", type=float)
    parser.add_argument("--verify-head", type=Path)
    parser.add_argument("--verify-threshold", type=float)
    parser.add_argument("--quant-bits", type=int, choices=(4, 8),
                        help="quantize the merged stock MLX backbone after loading")
    parser.add_argument("--run", default="jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778")
    parser.add_argument("--state-tokens", type=int, default=270)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--choices", type=int, default=3)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.engine == "nerqova-early" and (args.exit_head is None or args.exit_gap is None):
        parser.error("nerqova-early requires --exit-head and --exit-gap")
    if args.engine != "nerqova-early" and (args.exit_head is not None or args.exit_gap is not None or args.exit_gap_wide is not None):
        parser.error("exit options require nerqova-early")
    if args.exit_gap_wide is not None and args.exit_gap is None:
        parser.error("exit-gap-wide requires exit-gap")
    if (args.verify_head is None) != (args.verify_threshold is None) or (args.verify_head and args.engine != "nerqova-early"):
        parser.error("verifier options must be set together for nerqova-early")
    if args.quant_bits and args.engine != "kev-mlx":
        parser.error("quant-bits is a stock Kev MLX baseline only")
    if args.reps < 1 or args.warmups < 0 or args.state_tokens < 1 or args.questions < 1 or args.choices < 2:
        parser.error("reps, state-tokens and questions must be positive; warmups must be nonnegative")

    mx.set_cache_limit(1024 ** 3)
    if args.engine == "kev-mlx":
        checkpoint = Checkpoint(args.run)
        tok, model = checkpoint.load("mps", LoadOptions(backend="mlx"))
    else:
        checkpoint, tok, model = load_model(
            args.run, unmasked_branches=args.engine == "nerqova-unmasked",
            packed_delta=args.engine in ("nerqova-packed", "nerqova-early"),
            exit_head=args.exit_head, exit_gap=args.exit_gap, exit_gap_wide=args.exit_gap_wide,
            verify_head=args.verify_head, verify_threshold=args.verify_threshold,
        )
    if args.quant_bits:
        nn.quantize(model.lm, group_size=64, bits=args.quant_bits)
        mx.eval(model.lm.parameters())
        mx.clear_cache()
    example = workload(tok, args.state_tokens, args.questions, args.choices)
    request = SystemOneRequest.model_validate({
        "state": example["state"],
        "questions": {str(i): {"type": "choice", "instructions": q["instr"],
                               "criteria": {option: None for option in q["options"]}}
                      for i, q in enumerate(example["questions"])},
    })
    encoded = model.encode(tok, to_record(request)[0])
    server = Server(checkpoint, tok, model, "mps")
    response = server.answer(request)
    power_settings = subprocess.check_output(["pmset", "-g", "custom"], text=True)
    power_report = subprocess.check_output(["system_profiler", "SPPowerDataType"], text=True)
    modes = re.findall(r"powermode\s+(\d+)", power_settings)
    result = {
        "engine": args.engine,
        "quant_bits": args.quant_bits,
        "quant_group_size": 64 if args.quant_bits else None,
        "run": args.run,
        "checkpoint_revision": Path(checkpoint.path).name,
        "base_revision": checkpoint.meta.base_revision,
        "adapter_sha256": digest(checkpoint.file("adapter_model.safetensors")),
        "head_sha256": digest(checkpoint.file("head.pt")),
        "temperature": model.head.temperature,
        "exit_head_sha256": digest(args.exit_head) if args.exit_head else None,
        "exit_gap": args.exit_gap,
        "exit_gap_wide": args.exit_gap_wide,
        "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
        "verify_threshold": args.verify_threshold,
        "encoded_request_sha256": hashlib.sha256(json.dumps(encoded["ids"]).encode()).hexdigest(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "power": {"pmset_battery_mode": modes[0] if len(modes) > 0 else None,
                  "pmset_adapter_mode": modes[1] if len(modes) > 1 else None,
                  "reported_low_power": "Low Power Mode: Yes" in power_report,
                  "reported_high_power": "High Power Mode: Yes" in power_report},
        "state_tokens": encoded["seg"].count(0),
        "packed_tokens": len(encoded["ids"]),
        "questions": len(example["questions"]),
        "choices_per_question": args.choices,
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

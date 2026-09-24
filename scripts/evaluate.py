"""Score one engine with unchanged Kev weights on frozen development suites.

    uv run python scripts/evaluate.py --engine kev-mlx --out runs/kev-mlx-dev
    uv run python scripts/evaluate.py --engine nerqova --out runs/nerqova-dev

The checkpoint's served temperature is used. This script never reads locked test data.
"""

import argparse
import json
import subprocess
from pathlib import Path

from kev.benchmark import evaluate_records
from kev.checkpoint import Checkpoint, LoadOptions
from kev.predictors import LocalPredictor
from kev.suite import CONTEXT, digest, load_split, read_manifest, write_json

from nerqova.predictor import MLXPredictor


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def exits(path):
    with path.open() as source:
        records = [json.loads(line)["prediction"]["exit_stats"] for line in source]
    return {"records": len(records),
            "questions": sum(row["questions"] for row in records),
            "exited_questions": sum(row["exited"] for row in records),
            "complete_record_exits": sum(row["deferred"] == 0 for row in records),
            "verifier_vetoes": sum(row.get("verifier_vetoes", 0) for row in records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["kev-mlx", "nerqova", "nerqova-unmasked", "nerqova-packed", "nerqova-early"], required=True)
    parser.add_argument("--exit-head", type=Path)
    parser.add_argument("--exit-threshold", type=float)
    parser.add_argument("--verify-head", type=Path)
    parser.add_argument("--verify-threshold", type=float)
    parser.add_argument("--run", default="jaredpalmer/kev-4b@485ace8703592fcf405488b262449990824cfed1")
    parser.add_argument("--suite", type=Path, default=Path("evals/v7/decision-v7"))
    parser.add_argument("--transfer", type=Path, default=Path("evals/v4/transfer-v4"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.engine == "nerqova-early" and (args.exit_head is None or args.exit_threshold is None):
        parser.error("nerqova-early requires --exit-head and --exit-threshold")
    if args.engine != "nerqova-early" and (args.exit_head is not None or args.exit_threshold is not None):
        parser.error("exit options require nerqova-early")
    if (args.verify_head is None) != (args.verify_threshold is None) or (args.verify_head and args.engine != "nerqova-early"):
        parser.error("verifier options must be set together for nerqova-early")

    checkpoint = Checkpoint(args.run)
    suite_manifest = read_manifest(args.suite)
    transfer_manifest = read_manifest(args.transfer)
    args.out.mkdir(parents=True, exist_ok=False)
    predictor = (
        MLXPredictor(
            args.run, context=suite_manifest.get("context", CONTEXT),
            unmasked_branches=args.engine == "nerqova-unmasked",
            packed_delta=args.engine in ("nerqova-packed", "nerqova-early"),
            exit_head=args.exit_head, exit_threshold=args.exit_threshold,
            verify_head=args.verify_head, verify_threshold=args.verify_threshold,
        )
        if args.engine != "kev-mlx"
        else LocalPredictor(args.run, "mps", LoadOptions(backend="mlx"),
                            context=suite_manifest.get("context", CONTEXT))
    )

    development, _ = evaluate_records(
        load_split(args.suite, "development"), predictor, args.out / "development",
        heldout_sources=tuple(suite_manifest.get("holdout_sources", [])),
    )
    predictor.context = transfer_manifest.get("context", CONTEXT)
    transfer, _ = evaluate_records(
        load_split(args.transfer, "development"), predictor, args.out / "transfer",
        heldout_sources=tuple(transfer_manifest.get("holdout_sources", [])),
        skip_overlong=bool(transfer_manifest.get("eval_only")),
    )
    report = {
        "run": args.run,
        "engine": args.engine,
        "checkpoint": str(checkpoint.path),
        "base": checkpoint.meta.base,
        "base_revision": checkpoint.meta.base_revision,
        "adapter_sha256": digest(checkpoint.file("adapter_model.safetensors")),
        "head_sha256": digest(checkpoint.file("head.pt")),
        "served_temperature": predictor.temperature,
        "exit_head": str(args.exit_head) if args.exit_head else None,
        "exit_threshold": args.exit_threshold,
        "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
        "verify_threshold": args.verify_threshold,
        "exit_head_sha256": digest(args.exit_head) if args.exit_head else None,
        "exit_temperature": getattr(predictor.model, "exit_temperature", None),
        "code_sha": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain")),
        "suite_sha256": digest(args.suite / "manifest.json"),
        "transfer_suite_sha256": digest(args.transfer / "manifest.json"),
        "development": {"clean": development["clean"], "coverage": development["coverage"]},
        "transfer": {"clean": transfer["clean"], "coverage": transfer["coverage"],
                     "paired_flip": transfer["paired_flip"], "unknowable": transfer["unknowable"]},
        "locked_test_read": False,
    }
    if args.engine == "nerqova-early":
        report["exit_stats"] = {part: exits(args.out / part / "predictions.jsonl")
                                for part in ("development", "transfer")}
    write_json(args.out / "result.json", report)
    print(json.dumps({"engine": args.engine, "development": report["development"],
                      "transfer": report["transfer"]}, indent=2))


if __name__ == "__main__":
    main()

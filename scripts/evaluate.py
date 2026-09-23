"""Score one engine with unchanged Kev weights on frozen development suites.

    uv run python scripts/evaluate.py --engine kev-mlx --run jaredpalmer/kev-4b --out runs/kev-mlx-dev
    uv run python scripts/evaluate.py --engine nerqova --run jaredpalmer/kev-4b --out runs/nerqova-dev

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["kev-mlx", "nerqova"], required=True)
    parser.add_argument("--run", default="jaredpalmer/kev-4b")
    parser.add_argument("--suite", type=Path, default=Path("evals/v7/decision-v7"))
    parser.add_argument("--transfer", type=Path, default=Path("evals/v4/transfer-v4"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = Checkpoint(args.run)
    suite_manifest = read_manifest(args.suite)
    transfer_manifest = read_manifest(args.transfer)
    args.out.mkdir(parents=True, exist_ok=False)
    predictor = (
        MLXPredictor(args.run, context=suite_manifest.get("context", CONTEXT))
        if args.engine == "nerqova"
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
        "code_sha": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain")),
        "suite_sha256": digest(args.suite / "manifest.json"),
        "transfer_suite_sha256": digest(args.transfer / "manifest.json"),
        "development": {"clean": development["clean"], "coverage": development["coverage"]},
        "transfer": {"clean": transfer["clean"], "coverage": transfer["coverage"],
                     "paired_flip": transfer["paired_flip"], "unknowable": transfer["unknowable"]},
        "locked_test_read": False,
    }
    write_json(args.out / "result.json", report)
    print(json.dumps({"engine": args.engine, "development": report["development"],
                      "transfer": report["transfer"]}, indent=2))


if __name__ == "__main__":
    main()

"""Calibrate and score the MLX service path on frozen development partitions.

    uv run python scripts/evaluate.py --run runs/student35-2b-v7/00-trial-0/checkpoint \
      --out runs/student35-2b-v7-mlx

This script does not read the locked test partition.
"""

import argparse
import json
import subprocess
from pathlib import Path

from kev.benchmark import evaluate_records
from kev.metrics import fit_temperature
from kev.suite import CONTEXT, digest, load_split, read_manifest, write_json

from nerqova.predictor import MLXPredictor


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--suite", type=Path, default=Path("evals/v7/decision-v7"))
    parser.add_argument("--transfer", type=Path, default=Path("evals/v4/transfer-v4"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    predictor = MLXPredictor(args.run, context=read_manifest(args.suite)["context"])
    calibration, calibration_rows = evaluate_records(
        load_split(args.suite, "calibration"), predictor, args.out / "calibration"
    )
    temperature = fit_temperature(calibration_rows, aggregation="micro")
    predictor.set_temperature(temperature)

    development, _ = evaluate_records(
        load_split(args.suite, "development"), predictor, args.out / "development",
        heldout_sources=tuple(read_manifest(args.suite).get("holdout_sources", [])),
    )
    transfer_suite = read_manifest(args.transfer)
    predictor.context = transfer_suite.get("context", CONTEXT)
    transfer, _ = evaluate_records(
        load_split(args.transfer, "development"), predictor, args.out / "transfer",
        heldout_sources=tuple(transfer_suite.get("holdout_sources", [])),
        skip_overlong=bool(transfer_suite.get("eval_only")),
    )
    report = {
        "run": args.run,
        "backend": "mlx",
        "base": predictor.checkpoint.meta.base,
        "base_revision": predictor.checkpoint.meta.base_revision,
        "checkpoint": str(predictor.run),
        "adapter_sha256": digest(Path(predictor.run) / "adapter_model.safetensors"),
        "head_sha256": digest(Path(predictor.run) / "head.pt"),
        "code_sha": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain")),
        "suite_sha256": digest(args.suite / "manifest.json"),
        "transfer_suite_sha256": digest(args.transfer / "manifest.json"),
        "calibration_rows_sha256": digest(args.out / "calibration" / "rows.json"),
        "temperature": temperature,
        "calibration_coverage": calibration["coverage"],
        "development": {"clean": development["clean"], "coverage": development["coverage"]},
        "transfer": {"clean": transfer["clean"], "coverage": transfer["coverage"],
                     "paired_flip": transfer["paired_flip"], "unknowable": transfer["unknowable"]},
        "locked_test_read": False,
    }
    write_json(args.out / "result.json", report)
    print(json.dumps({"temperature": temperature, "development": report["development"],
                      "transfer": report["transfer"]}, indent=2))


if __name__ == "__main__":
    main()

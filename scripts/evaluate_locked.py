"""One confirmatory locked-test run after the runtime and gate are frozen."""

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
    parser.add_argument("--allow-test", action="store_true", help="required for one final locked-test read")
    parser.add_argument("--engine", choices=("kev-mlx", "nerqova-early"), required=True)
    parser.add_argument("--run", default="jaredpalmer/kev-4b")
    parser.add_argument("--suite", type=Path, default=Path("evals/v7/decision-v7"))
    parser.add_argument("--transfer", type=Path, default=Path("evals/v4/transfer-v4"))
    parser.add_argument("--exit-head", type=Path)
    parser.add_argument("--exit-threshold", type=float)
    parser.add_argument("--verify-head", type=Path)
    parser.add_argument("--verify-threshold", type=float)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.allow_test:
        parser.error("locked test requires explicit --allow-test")
    gate = (args.exit_head, args.exit_threshold, args.verify_head, args.verify_threshold)
    if args.engine == "nerqova-early" and any(value is None for value in gate):
        parser.error("nerqova-early requires both heads and thresholds")
    if args.engine == "kev-mlx" and any(value is not None for value in gate):
        parser.error("head options require nerqova-early")
    if git("status", "--porcelain"):
        parser.error("locked test requires a clean committed checkout")

    suite = read_manifest(args.suite)
    transfer = read_manifest(args.transfer)
    checkpoint = Checkpoint(args.run)
    args.out.mkdir(parents=True, exist_ok=False)
    predictor = (
        LocalPredictor(args.run, "mps", LoadOptions(backend="mlx"),
                       context=suite.get("context", CONTEXT))
        if args.engine == "kev-mlx"
        else MLXPredictor(args.run, context=suite.get("context", CONTEXT),
                          packed_delta=True, exit_head=args.exit_head,
                          exit_threshold=args.exit_threshold,
                          verify_head=args.verify_head,
                          verify_threshold=args.verify_threshold)
    )
    decision, _ = evaluate_records(
        load_split(args.suite, "test", allow_test=True), predictor,
        args.out / "decision_test",
        heldout_sources=tuple(suite.get("holdout_sources", [])),
    )
    predictor.context = transfer.get("context", CONTEXT)
    other, _ = evaluate_records(
        load_split(args.transfer, "test", allow_test=True), predictor,
        args.out / "transfer_test",
        heldout_sources=tuple(transfer.get("holdout_sources", [])),
        skip_overlong=bool(transfer.get("eval_only")),
    )
    report = {
        "engine": args.engine,
        "code_sha": git("rev-parse", "HEAD"),
        "code_dirty": False,
        "run": args.run,
        "checkpoint_revision": Path(checkpoint.path).name,
        "base_revision": checkpoint.meta.base_revision,
        "adapter_sha256": digest(checkpoint.file("adapter_model.safetensors")),
        "head_sha256": digest(checkpoint.file("head.pt")),
        "exit_head_sha256": digest(args.exit_head) if args.exit_head else None,
        "exit_threshold": args.exit_threshold,
        "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
        "verify_threshold": args.verify_threshold,
        "suite_sha256": digest(args.suite / "manifest.json"),
        "transfer_suite_sha256": digest(args.transfer / "manifest.json"),
        "decision": {"clean": decision["clean"], "coverage": decision["coverage"]},
        "transfer": {"clean": other["clean"], "coverage": other["coverage"]},
        "locked_test_read": True,
    }
    write_json(args.out / "result.json", report)
    print(json.dumps({"engine": args.engine,
                      "decision": {key: decision["clean"][key] for key in ("acc", "brier", "ece", "nll")},
                      "transfer": {key: other["clean"][key] for key in ("acc", "brier", "ece", "nll")}}))


if __name__ == "__main__":
    main()

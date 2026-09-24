"""Export served full-model probabilities for the training split of one checkpoint.

The JSONL sidecar permits resume after interruption. The final gzip is the
target format consumed by early_exit_features.py. No held-out split is read.
"""

import argparse
import gzip
import json
from pathlib import Path

import mlx.core as mx

from kev.data import materialize
from kev.checkpoint import Checkpoint
from kev.suite import CONTEXT, digest, load_split, read_manifest

from nerqova.checkpoint import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--suite", type=Path, default=Path("evals/v7/decision-v7"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.out.exists():
        parser.error("final target file already exists")

    manifest = read_manifest(args.suite)
    context = manifest.get("context", CONTEXT)
    records = load_split(args.suite, "train")
    if args.limit is not None:
        records = records[:args.limit]
    revision = Path(Checkpoint(args.run).path).name
    sidecar = args.out.with_suffix("").with_suffix(".jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sidecar_meta = {"run": args.run, "checkpoint_revision": revision,
                    "suite_sha256": digest(args.suite / "manifest.json"),
                    "rows_sha256": digest(args.suite / "train.jsonl"),
                    "records": len(records)}

    completed = []
    if sidecar.exists():
        with sidecar.open() as source:
            if json.loads(source.readline()).get("_meta") != sidecar_meta:
                raise ValueError("teacher sidecar has different checkpoint or training records")
            for index, line in enumerate(source):
                row = json.loads(line)
                if index >= len(records) or row["id"] != records[index]["_meta"]["id"]:
                    raise ValueError("teacher sidecar does not match this suite and split")
                completed.append(row)
    else:
        sidecar.write_text(json.dumps({"_meta": sidecar_meta}) + "\n")
    mx.set_cache_limit(1024 ** 3)
    checkpoint, tok, model = load_model(args.run, packed_delta=True)
    if Path(checkpoint.path).name != revision:
        raise ValueError("checkpoint revision changed during target export")
    with sidecar.open("a") as destination:
        for index in range(len(completed), len(records)):
            record = materialize(records[index])
            encoded = model.encode(
                tok, record, max_state=context["max_state"],
                max_branch=context["max_branch"], strict=True,
            )
            if len(encoded["ids"]) > context["max_packed"]:
                raise ValueError(f"packed request exceeds context: {records[index]['_meta']['id']}")
            probabilities = model.probs(encoded)
            if len(probabilities) != len(record["questions"]):
                raise ValueError("model returned the wrong question count")
            targets = {}
            for question, values in zip(record["questions"], probabilities, strict=True):
                vector = values.tolist()
                if len(vector) != len(question["keys"]):
                    raise ValueError(f"model returned the wrong option count for {question['qid']}")
                targets[question["qid"]] = dict(zip(question["keys"], vector, strict=True))
            row = {"id": records[index]["_meta"]["id"], "targets": targets}
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
            completed.append(row)
            if (index + 1) % 250 == 0 or index + 1 == len(records):
                destination.flush()
                print(json.dumps({"completed": index + 1, "total": len(records),
                                  "active_mib": round(mx.get_active_memory() / 2**20),
                                  "cache_mib": round(mx.get_cache_memory() / 2**20)}), flush=True)

    document = {
        "_meta": {
            "teacher": args.run,
            "checkpoint_revision": Path(checkpoint.path).name,
            "base_revision": checkpoint.meta.base_revision,
            "teacher_temperature": model.head.temperature,
            "suite": str(args.suite),
            "suite_sha256": digest(args.suite / "manifest.json"),
            "partition": "train",
            "rows_sha256": digest(args.suite / "train.jsonl"),
            "records": len(completed),
            "questions": sum(len(row["targets"]) for row in completed),
            "readout": "served full-model probability vectors",
        },
        "targets": {row["id"]: row["targets"] for row in completed},
    }
    args.out.write_bytes(gzip.compress(json.dumps(document, separators=(",", ":")).encode()))
    print(json.dumps({"out": str(args.out), "records": len(completed),
                      "questions": document["_meta"]["questions"],
                      "sha256": digest(args.out)}), flush=True)


if __name__ == "__main__":
    main()

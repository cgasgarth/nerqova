"""Export frozen training/calibration anchors from a partial Kev-4B forward.

Each shard contains only normalized decision and option vectors, never token
states or model weights. Development and locked test are excluded by design.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from kev.data import materialize
from kev.suite import CONTEXT, digest, load_split, read_manifest
from nerqova.checkpoint import load_model
from nerqova.early_exit import anchor_vectors, branch_at, encoded_rows, prefix_at


def flush(path, items):
    q_vectors, option_vectors, targets, labels, offsets, record_numbers = [], [], [], [], [0], []
    for record_number, examples in items:
        for decide, options, teacher, label in examples:
            q_vectors.append(decide)
            option_vectors.extend(options)
            targets.extend(teacher)
            labels.append(label)
            record_numbers.append(record_number)
            offsets.append(offsets[-1] + len(options))
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, decide=np.stack(q_vectors), options=np.stack(option_vectors),
             targets=np.asarray(targets, dtype=np.float32),
             labels=np.asarray(labels, dtype=np.int16),
             offsets=np.asarray(offsets, dtype=np.int32),
             record_numbers=np.asarray(record_numbers, dtype=np.int32))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="jaredpalmer/kev-4b")
    parser.add_argument("--suite", type=Path, default=Path("evals/v7/decision-v7"))
    parser.add_argument("--teacher", type=Path, default=Path("runs/weights/kev4b-v7-train-teacher.json.gz"))
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--layer", type=int, choices=(8, 16, 24), required=True)
    parser.add_argument("--shard-records", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shard_records < 1 or (args.limit is not None and args.limit < 1):
        parser.error("shard-records and limit must be positive")

    manifest = read_manifest(args.suite)
    context = manifest.get("context", CONTEXT)
    records = load_split(args.suite, args.split)
    if args.limit is not None:
        records = records[:args.limit]
    teacher = None
    teacher_sha256 = None
    if args.split == "train":
        raw_teacher = gzip.decompress(args.teacher.read_bytes())
        teacher_sha256 = hashlib.sha256(raw_teacher).hexdigest()
        teacher_doc = json.loads(raw_teacher)
        if teacher_doc["_meta"]["suite_sha256"] != digest(args.suite / "manifest.json"):
            raise ValueError("teacher targets refer to another suite")
        teacher = teacher_doc["targets"]
    checkpoint, tok, model = load_model(args.run, packed_delta=True)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"run": args.run, "checkpoint_revision": Path(checkpoint.path).name,
                "base_revision": checkpoint.meta.base_revision,
                "suite_sha256": digest(args.suite / "manifest.json"),
                "split": args.split, "layer": args.layer, "records": len(records),
                "shard_records": args.shard_records,
                "teacher_sha256": teacher_sha256}
    meta_path = args.out / "manifest.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise ValueError("existing feature directory has different inputs")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    for start in range(0, len(records), args.shard_records):
        path = args.out / f"{start:06d}.npz"
        if path.exists():
            continue
        items = []
        for index in range(start, min(start + args.shard_records, len(records))):
            record = records[index]
            rec = materialize(record)
            enc = model.encode(tok, rec, max_state=context["max_state"],
                               max_branch=context["max_branch"], strict=True)
            if len(enc["ids"]) > context["max_packed"]:
                raise ValueError(f"packed request exceeds context: {record['_meta']['id']}")
            state_ids, rows, chunk = encoded_rows(model, enc)
            _, cache = prefix_at(model, state_ids, args.layer)
            vectors = []
            for offset in range(0, len(rows), chunk):
                part = rows[offset:offset + chunk]
                hidden, _ = branch_at(model, part, cache, args.layer)
                vectors.extend(anchor_vectors(model, hidden, part))
            calibration_probs = model.probs(enc) if args.split == "calibration" else None
            examples = []
            for q_index, (q, matrix) in enumerate(zip(rec["questions"], vectors, strict=True)):
                target = ([teacher[record["_meta"]["id"]][q["qid"]][key] for key in q["keys"]]
                          if teacher is not None else calibration_probs[q_index].tolist())
                examples.append((matrix[0], matrix[1:], target, q["label"]))
            items.append((index, examples))
        flush(path, items)
        print(f"exported {min(start + args.shard_records, len(records))}/{len(records)} records", flush=True)


if __name__ == "__main__":
    main()

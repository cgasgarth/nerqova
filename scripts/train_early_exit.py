"""Train a pointer head on frozen partial-backbone anchors.

Training uses only the train split. The separate calibration split sets the
served temperature and records teacher agreement for exit-gate design.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kev.checkpoint import Checkpoint
from nerqova.early_head import PairHead


def shard_paths(directory):
    paths = sorted(directory.glob("*.npz"))
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = [directory / f"{start:06d}.npz"
                for start in range(0, manifest["records"], manifest["shard_records"])]
    if paths != expected:
        raise ValueError(f"feature shards are incomplete or unexpected in {directory}")
    return paths


def batches(path, size, *, shuffle, rng):
    with np.load(path) as data:
        order = list(range(len(data["labels"])))
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), size):
            selected = order[start:start + size]
            counts = [int(data["offsets"][i + 1] - data["offsets"][i]) for i in selected]
            width = max(counts)
            decide = np.stack([data["decide"][i] for i in selected])
            options = np.zeros((len(selected), width, decide.shape[1]), dtype=np.float16)
            targets = np.zeros((len(selected), width), dtype=np.float32)
            for j, i in enumerate(selected):
                begin, end = data["offsets"][i:i + 2]
                options[j, :counts[j]] = data["options"][begin:end]
                targets[j, :counts[j]] = data["targets"][begin:end]
            yield decide, options, targets, np.asarray([data["labels"][i] for i in selected]), counts, [int(data["record_numbers"][i]) for i in selected]


def forward(head, batch, device):
    decide, options, targets, labels, counts, records = batch
    decide = torch.from_numpy(decide.astype(np.float32)).to(device)
    options = torch.from_numpy(options.astype(np.float32)).to(device)
    logits = head(decide, options)
    mask = torch.arange(options.shape[1], device=device)[None, :] >= torch.tensor(counts, device=device)[:, None]
    return logits.masked_fill(mask, -1e4), torch.from_numpy(targets).to(device), torch.from_numpy(labels.astype(np.int64)).to(device)


@torch.no_grad()
def collect_calibration(head, paths, batch_size, device):
    head.eval()
    rng = random.Random(0)
    logits_rows, teacher_rows, labels, records = [], [], [], []
    for path in paths:
        for batch in batches(path, batch_size, shuffle=False, rng=rng):
            logits, targets, y = forward(head, batch, device)
            for index, count in enumerate(batch[4]):
                logits_rows.append(logits[index, :count].cpu().numpy())
                teacher_rows.append(targets[index, :count].cpu().numpy())
            labels.extend(y.cpu().tolist())
            records.extend(batch[5])
    return logits_rows, teacher_rows, np.asarray(labels), np.asarray(records)


def calibration_report(logits_rows, teacher_rows, labels):
    # Fit one scalar temperature on held-out labels. Choice is unchanged.
    best = None
    for temperature in np.exp(np.linspace(math.log(0.5), math.log(8.0), 121)):
        probs = [torch.softmax(torch.from_numpy(z) / temperature, -1).numpy() for z in logits_rows]
        nll = float(np.mean([-math.log(max(float(p[y]), 1e-12)) for p, y in zip(probs, labels)]))
        if best is None or nll < best[0]:
            best = (nll, float(temperature), probs)
    nll, temperature, probs = best
    student_choices = np.asarray([int(np.argmax(p)) for p in probs])
    teacher_choices = np.asarray([int(np.argmax(p)) for p in teacher_rows])
    return {"temperature": temperature, "nll": nll,
            "accuracy": float(np.mean(student_choices == labels)),
            "teacher_accuracy": float(np.mean(teacher_choices == labels)),
            "teacher_agreement": float(np.mean(student_choices == teacher_choices)),
            "questions": len(labels)}, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778")
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or not 0 <= args.teacher_weight <= 1:
        parser.error("epochs, batch-size and learning-rate must be positive; teacher-weight must be in [0, 1]")
    train_meta = json.loads((args.train_dir / "manifest.json").read_text())
    cal_meta = json.loads((args.calibration_dir / "manifest.json").read_text())
    for key in ("run", "checkpoint_revision", "base_revision", "suite_sha256", "layer", "max_state_tokens"):
        if train_meta[key] != cal_meta[key]:
            raise ValueError(f"feature inputs disagree on {key}")
    if train_meta["split"] != "train" or cal_meta["split"] != "calibration":
        raise ValueError("expected separate train and calibration features")
    checkpoint = Checkpoint(args.run)
    torch.manual_seed(1827)
    head = PairHead()
    device = torch.device(args.device)
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    rng = random.Random(1827)
    train_paths = shard_paths(args.train_dir)
    calibration_paths = shard_paths(args.calibration_dir)
    history = []
    best = None
    best_epoch = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        rng.shuffle(train_paths)
        head.train()
        total_loss, seen = 0.0, 0
        for path in train_paths:
            for batch in batches(path, args.batch_size, shuffle=True, rng=rng):
                logits, teacher, labels = forward(head, batch, device)
                hard = F.cross_entropy(logits, labels)
                soft = -(teacher * F.log_softmax(logits, -1)).sum(-1).mean()
                loss = (1 - args.teacher_weight) * hard + args.teacher_weight * soft
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                seen += len(labels)
        rows, teachers, labels, records = collect_calibration(head, calibration_paths, args.batch_size, device)
        report, probabilities = calibration_report(rows, teachers, labels)
        report.update(epoch=epoch + 1, train_loss=total_loss / seen,
                      train_questions=seen, teacher_weight=args.teacher_weight,
                      head_type="pair")
        history.append(report)
        print(json.dumps(report), flush=True)
        if best is None or report["nll"] < best["nll"]:
            best = report
            best_epoch = epoch + 1
            torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        "head_type": "pair",
                        "layer": train_meta["layer"], "temperature": report["temperature"],
                        "train_manifest": train_meta, "calibration_manifest": cal_meta,
                        "calibration": report}, args.out)
            offsets = np.cumsum([0] + [len(p) for p in probabilities])
            np.savez(args.out.with_suffix(".calibration.npz"),
                     student=np.concatenate(probabilities), teacher=np.concatenate(teachers),
                     offsets=offsets, labels=labels, record_numbers=records)
        if epoch + 1 - best_epoch >= 2:
            break
    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"best": best, "artifact": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()

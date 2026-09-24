# Nerqova exit-head research

Nerqova trains a partial-backbone readout for the pinned public 4B decision
checkpoint. Its backbone, adapter, and full pointer head remain unchanged.
These heads add an optional conditional exit; they are not the default runtime
path because the selected quality-safe gate did not improve the fixed complete
request benchmark.

## Assets and provenance

The v0.2.0 release contains these files. Keep them under ignored `runs/weights/`.

| Asset | Purpose | SHA-256 |
| --- | --- | --- |
| `nerqova-exit16-r8.pt` | Layer-16 conditional readout | `d2d49c05c9f4b15d5a1e16ae4a9fc0ddd12d060ff7f62a9eadec70f1e720c2fc` |
| `nerqova-exit8-r8.pt` | Layer-8 research readout; not used by the selected gate | `d729281dbf32158e5daaf1cbf15ac6afea822dd38aefe9a8ce7d0b9ed8cb54bd` |
| `nerqova-teacher-r8.json.gz` | Training-split full-model probability targets | `6c131dcb2c16db4b18778f74f57e90844a3c86fa02639f572b35574b544a5106` |

Both heads require checkpoint revision
`1da696f7938f77c4cdf5471e92fd342baff41778`, whose base is
`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`.
The loader checks the checkpoint revision. The teacher targets contain served
probabilities from the same full checkpoint for 12,576 decision-v7 training
records (15,576 questions). They contain no development or locked-test rows.

Training extracted normalized decision and option vectors after layers 8 and
16. A 384-wide nonlinear pair head learned from hard labels and full-model
probabilities with equal loss weight. Epoch and temperature were selected on
968 separate calibration records (1,148 questions). Layer 8 reached 62.46%
calibration accuracy, so it is not used as a verifier. Layer 16 reached 85.28%
against 87.54% for the full model before gating.

## Selected gate

For each question, compute `ln(top_probability / runner_up_probability)` from
the calibrated layer-16 readout. Exit when the gap is at least **5.0** for
2–13 options, or **2.5** for 14 or more options. A 15-option distribution
with a 60% winner and a 2.9% runner-up passes the wide-choice rule even
though the top probability is below 90%. Other questions continue through
all 32 layers and use the original full pointer head.

The thresholds came from calibration, then passed the frozen decision-v7,
transfer-v4, and documents-v1 development suites. Calibration had zero
full-model choice changes at this gate and accepted 25.4% of complete records.
On development, the gate changed zero top choices across 1,468 decision-v7,
764 transfer-v4, and 920 documents-v1 questions. It exited on 364 / 1,204,
95 / 764, and 11 / 568 complete records, respectively. Accuracy matched the
full model. Brier and ECE stayed within the predeclared limits; see
[the quality table](../docs/latest-performance.md). A state longer than the
384-token training context always uses the full model. These finite suites
cannot guarantee choice identity on new inputs.

The selected gate was **slower than the default packed runtime** on the fixed
five-question and 15-option fixtures. Use it as a research option for workloads
where its exits and calibration are useful; measure your own request mix. It did
not pass the predeclared speed gate, so this head was not advanced to a locked
test or made the default.

```bash
mkdir -p runs/weights
gh release download v0.2.0 --repo cgasgarth/nerqova \
  --pattern 'nerqova-*.pt' --dir runs/weights
uv run --no-dev python -m nerqova.serve \
  --run jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778 \
  --packed-delta --exit-head runs/weights/nerqova-exit16-r8.pt \
  --exit-gap 5 --exit-gap-wide 2.5 --port 8009
```

## Reproduce the training path

```bash
uv run python scripts/export_teacher_targets.py \
  --run jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778 \
  --out runs/new-checkpoint/teacher-train.json.gz
uv run python scripts/early_exit_features.py --split train --layer 16 \
  --teacher runs/new-checkpoint/teacher-train.json.gz \
  --out runs/new-checkpoint/train-l16
uv run python scripts/early_exit_features.py --split calibration --layer 16 \
  --out runs/new-checkpoint/calibration-l16
uv run python scripts/train_early_exit.py \
  --train-dir runs/new-checkpoint/train-l16 \
  --calibration-dir runs/new-checkpoint/calibration-l16 \
  --out runs/new-checkpoint/head-l16.pt
uv run python scripts/select_exit_gate.py runs/new-checkpoint/head-l16.pt \
  --wide-gap 2.5 --out runs/new-checkpoint/gates.json
```

Repeat feature extraction and training with layer 8 only if studying an earlier
readout. The repo uses [Kev](https://github.com/jaredpalmer/kev) checkpoint
format and frozen suites as research inputs; see [NOTICE](../NOTICE).

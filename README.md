# Nerqova

A fast, local decision model that serves Kev's typed System One API. This repository owns the Apple Silicon scorer, serving entry point, student experiments, and validation. It uses a pinned [Kev](https://github.com/jaredpalmer/kev) revision for encoding, training, evaluation, and the API contract.

**Status:** research. No student checkpoint has passed the release quality gate yet. The latency of an untrained architecture is not evidence of decision quality.

## What It Does

One state and a set of typed questions go in. Each question gets a probability distribution over its options in one scoring request. The scorer runs a pretrained Qwen3 or Qwen3.5 text backbone with a trained LoRA adapter and pointer head. It reuses a state prefix while keeping question branches isolated. It does not generate an answer token by token.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/nerqova/` | MLX scoring, checkpoint loading, and HTTP entry point |
| `evals/` | Checksummed Kev suite manifests and the pinned date/unknowable training delta |
| `experiments/` | Pinned student training configurations |
| `scripts/` | Model-only latency measurement and training-only teacher targets |
| `tests/` | Fast contract checks; model-backed parity is opt-in |
| `docs/model-cards/` | Model card requirements; released checkpoints get their own card |
| `evidence/` | Small, versioned benchmark reports used to define the comparison |

The Kev source is pinned to commit `557598fced1dada75dfbf36ed144dce309ac6ceb`. Missing frozen suite partitions are fetched from `jaredpalmer/kev-suites` at the revision pinned by Kev and verified against the manifests. The locked test partition requires an explicit `--allow-test` after a finalist is selected.

## Setup

On Apple Silicon with Python 3.13:

```bash
uv sync --extra serve --group dev
uv run python -m pytest tests -q -m 'not model'
```

Check a training plan without starting a GPU job:

```bash
uv run python -m kev.experiment \
  --suite evals/v7/decision-v7 \
  --plan experiments/student35-2b-v7.json \
  --out runs/student35-2b-v7 \
  --transfer evals/v4/transfer-v4 \
  --dry-run
```

The same command without `--dry-run` trains and evaluates the plan. Local training can occupy the GPU for hours. Kev's Modal workflow can run the plan on an H100 when a Modal account is connected.

The trainer's study report uses Kev's PyTorch evaluator. Check served MLX probabilities with the repo's evaluator after training:

```bash
uv run python scripts/evaluate.py --run runs/student35-2b-v7/00-trial-0/checkpoint --out runs/student35-2b-v7-mlx
```

It fits the temperature on the frozen calibration partition, then scores development and transfer development through the MLX path. It does not read the locked test.

Serve a trained Kev-format checkpoint and measure model-only latency:

```bash
uv run python -m nerqova.serve --run runs/student35-2b-v7/00-trial-0/checkpoint --port 8009
uv run python scripts/bench_decisions.py --run runs/student35-2b-v7/00-trial-0/checkpoint --out runs/latency.json
```

`scripts/bench_decisions.py` starts with an already encoded request and returns fully materialized CPU probability vectors. Complete request and HTTP timing must be measured separately before a release claim.

## Release Gate

A student must match or improve the current Kev-4B service on held-out accuracy and served probability quality, then deliver at least **2× lower complete decision latency** on the same M5 Pro workload. The initial transfer-v4 development reference, scored through MLX on 656 knowable questions, is **0.800 accuracy, 0.264 Brier, and 0.036 ECE**. Check confident errors, unknowable controls, policy pairs, option order, and question isolation. Fit the student temperature on the calibration partition. Open the locked test only for the selected finalist and report paired uncertainty.

The [baseline report](evidence/kev4b-transfer-v4-development.json) records this development measurement. Its checkpoint request was not pinned to a Hub commit, so it is a research reference. Rerun both models with exact checkpoint and code revisions before a release claim.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for Kev attribution.

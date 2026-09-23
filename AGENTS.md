# Nerqova

This repo owns the fast student scorer and its evidence. The pinned Kev dependency owns the typed API, encoder, trainer, frozen-suite loader, and core metrics. Extend those contracts only when a measured need requires it.

## Commands

- Environment: `uv sync --extra serve --group dev` on Apple Silicon, Python 3.13.
- Unit tests: `uv run python -m pytest tests -q -m 'not model'`.
- Model tests: `uv run python -m pytest tests -q -m model` on a Mac with model weights; do not run them in ordinary CI.
- Plan check: `uv run python -m kev.experiment --suite evals/v7/decision-v7 --plan experiments/student35-2b-v7.json --out runs/student35-2b-v7 --dry-run`.
- Served evaluation: `uv run python scripts/evaluate.py --run <checkpoint> --out runs/<name>-mlx`.
- Serve: `uv run python -m nerqova.serve --run <checkpoint> --port 8009`.
- Model latency: `uv run python scripts/bench_decisions.py --run <checkpoint> --out runs/latency.json`.

## Evidence Rules

- Keep exact source revision, dataset manifest hash, checkpoint revision, adapter/head hashes, temperature, device, and runtime versions with each result.
- Use only the frozen training partition for training and teacher targets. Select on development and fit calibration on its reserved partition. Read the locked test once for a selected finalist.
- Compare the exported MLX scorer with the current Kev-4B *served* probabilities and metrics. A fast untrained backbone does not pass the quality gate.
- Report model-only, complete local decision, and HTTP latency as separate measurements. Use the same request and hardware for candidate and baseline.
- Preserve independent question branches, option order, and the pristine state-prefix cache. Verify repeated hits, reordered questions, unequal branch lengths, and same-length different states before release.
- Keep large checkpoints and generated partitions under `runs/` or ignored `evals/` paths. Publish only checked results and a model card for a checkpoint that passes the gate.

# Nerqova

A model-specific Apple Silicon runtime for Kev-style decisions. The first target is the **same released Kev-4B checkpoint**, with faster hardware execution than the stock MLX path and matching probability vectors. It uses Kev's typed System One API and frozen evaluation suites through a pinned source revision.

**Status:** research. The current Nerqova path uses MLX for the backbone and a native MLX pointer head. On one short fixed request it is only about 1% faster than stock Kev MLX. Custom model-specific Metal work is next; there is no large same-weight speed claim yet.

## Goal

One state and a set of typed questions go in. A probability distribution for each question comes out in one scoring pass. No answer-token loop runs. Nerqova aims for at least 2× lower **complete decision latency** than stock Kev-4B MLX on the same M5 Pro request, with the same checkpoint weights, option choices, and close served probabilities. The [performance plan](docs/performance.md) explains which [Husky](https://husky.underdog.ai/) hardware ideas fit this workload.

## Repository layout

| Path | Purpose |
|---|---|
| `src/nerqova/` | Checkpoint loader, Apple Silicon scorer, and System One server |
| `scripts/bench_decisions.py` | Same-request model latency for stock Kev MLX and Nerqova |
| `scripts/evaluate.py` | Frozen development-suite quality for either engine |
| `tests/` | Contract and checkpoint-backed parity checks |
| `evals/` | Checksummed Kev suite manifests |
| `evidence/` | Versioned baseline report |

The Kev source is pinned to commit `557598fced1dada75dfbf36ed144dce309ac6ceb`. Kev suite partitions are fetched from the dataset revision in that source and checked against the manifests. [NOTICE](NOTICE) records Kev attribution.

## Set up

On Apple Silicon with Python 3.13:

```bash
uv sync --extra serve --group dev
uv run python -m pytest tests -q -m 'not model'
```

Measure each engine in its own quiet run:

```bash
uv run python scripts/bench_decisions.py --engine kev-mlx --run jaredpalmer/kev-4b --out runs/kev-mlx-latency.json
uv run python scripts/bench_decisions.py --engine nerqova --run jaredpalmer/kev-4b --out runs/nerqova-latency.json
```

The reports include checkpoint and input hashes, runtime versions, fully materialized probability vectors, and new-state and cached-state model times. They exclude encoding, response formatting, and HTTP. The release gate requires complete local and HTTP times too.

Use `scripts/bench_complete.py` for complete local decision time.

Compare two complete local reports with `uv run python scripts/compare_engines.py --complete runs/kev-mlx-complete.json runs/nerqova-complete.json`.

Evaluate the served probabilities on frozen development partitions:

```bash
uv run python scripts/evaluate.py --engine kev-mlx --out runs/kev-mlx-development
uv run python scripts/evaluate.py --engine nerqova --out runs/nerqova-development
```

Serve the same checkpoint with Nerqova's scorer:

```bash
uv run python -m nerqova.serve --run jaredpalmer/kev-4b --port 8009
```

The transfer-v4 development [baseline report](evidence/kev4b-transfer-v4-development.json) records 0.800 accuracy, 0.264 Brier, and 0.036 ECE on 656 knowable questions. Its Hub checkpoint request was not pinned to a commit, so rerun the comparison with pinned revisions before a release claim. The locked test remains closed until a final runtime has passed development checks.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

# Nerqova

A model-specific Apple Silicon runtime for Kev-style decisions. The first target is the **same released Kev-4B checkpoint**. A packed Metal recurrence preserves the previous scorer's probabilities. An optional trained early exit uses the same backbone and continues uncertain questions through all layers. It uses Kev's typed System One API and frozen evaluation suites through a pinned source revision.

**Status:** research. The default Nerqova path is effectively tied with stock Kev MLX. The optional packed Metal DeltaNet kernel has a measured same-weight gain. The [conditional exit heads](models/README.md) exceeded **2× median complete-request speed** on the fixed M5 Pro workload in clean local and HTTP runs. Their performance varies with how many questions defer; the varied decision-v7 median improved about **1.09×**. See the [exact-SHA evidence](evidence/conditional-exit-e23ddd3.json).

At clean commit `601727c`, the packed path was about **6% faster** than stock Kev MLX for complete local and HTTP decisions on a fixed M5 Pro request, with zero choice flips across the frozen development suites. See the [measurement and limits](docs/performance.md#packed-metal-result) and [evidence](evidence/packed-metal-601727c.json).

## Goal

One state and a set of typed questions go in. A probability distribution for each question comes out without an answer-token loop. Nerqova aims for at least 2× lower **complete decision latency** than stock Kev-4B MLX on the same M5 Pro request. The packed path keeps the same weights and close probabilities. The conditional exit keeps the same backbone but uses a trained readout and must pass held-out accuracy and calibration checks. The [performance plan](docs/performance.md) explains which [Husky](https://husky.underdog.ai/) hardware ideas fit this workload.

## Repository layout

| Path | Purpose |
|---|---|
| `src/nerqova/` | Checkpoint loader, Apple Silicon scorer, packed Metal recurrence, and System One server |
| `scripts/bench_decisions.py` | Same-request model latency for stock Kev MLX and Nerqova |
| `scripts/bench_complete.py`, `scripts/bench_http.py` | Complete local and HTTP decision timing |
| `scripts/evaluate.py` | Frozen development-suite quality for either engine |
| `scripts/early_exit_features.py`, `scripts/train_early_exit.py` | Same-backbone exit-head research on frozen train and calibration splits |
| `models/` | Trained-head model card and release-asset provenance |
| `tests/` | Contract and checkpoint-backed parity checks |
| `evals/` | Checksummed Kev suite manifests |
| `evidence/` | Versioned baseline report |
| `third_party/` | Required MIT license notice for the adapted MLX-LM kernel |

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
uv run python scripts/bench_decisions.py --engine nerqova-packed --run jaredpalmer/kev-4b --out runs/nerqova-packed-latency.json
```

The reports include checkpoint and input hashes, runtime versions, fully materialized probability vectors, and new-state and cached-state model times. They exclude encoding, response formatting, and HTTP. The release gate requires complete local and HTTP times too.

Use `scripts/bench_complete.py` for complete local decision time.

Compare two complete local reports with `uv run python scripts/compare_engines.py --complete runs/kev-mlx-complete.json runs/nerqova-packed-complete.json`.

Evaluate the served probabilities on frozen development partitions:

```bash
uv run python scripts/evaluate.py --engine kev-mlx --out runs/kev-mlx-development
uv run python scripts/evaluate.py --engine nerqova-packed --out runs/nerqova-packed-development
```

Serve the same checkpoint with Nerqova's scorer:

```bash
uv run python -m nerqova.serve --run jaredpalmer/kev-4b --packed-delta --port 8009
```

Use the conditional exit with its [pinned heads and gate](models/README.md):

```bash
mkdir -p runs/weights
gh release download v0.1.0 --repo cgasgarth/nerqova --pattern 'kev4b-*' --dir runs/weights
uv run python -m nerqova.serve --run jaredpalmer/kev-4b --packed-delta \
  --exit-head runs/weights/kev4b-exit16-pair.pt --exit-threshold 0.90 \
  --verify-head runs/weights/kev4b-verify8-pair.pt --verify-threshold 0.26
```

The [conditional-exit evidence](evidence/conditional-exit-e23ddd3.json) records the pinned checkpoint and input hashes, clean local and HTTP samples, and the one-time locked comparison. It found **zero choice flips across 2,204 locked questions** with full coverage. Accuracy and Brier matched or improved stock Kev; ECE was slightly higher. The [performance report](docs/performance.md#conditional-exit-candidate) gives the exact values and limits.


## License

Apache-2.0 for Nerqova's original code. The packed kernel adapts MIT-licensed MLX-LM code. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [the MLX-LM license](third_party/mlx_lm_LICENSE).

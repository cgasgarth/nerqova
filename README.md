# Nerqova

Nerqova serves Kev-style decisions on Apple Silicon. It takes one state and a set
of typed questions, then returns an option probability distribution for each
question. It uses the released **Kev-4B checkpoint** and the same System One
HTTP contract as Kev.

The runtime has two speed paths:

1. **Packed Metal DeltaNet:** a model-specific GPU recurrence assigns eight
   value rows to each SIMD group. It keeps the checkpoint weights and matches
   the prior recurrence output on the tested Kev-4B shapes.
2. **Verified early exit:** trained readouts at layers 8 and 16 let a confident
   question stop before all 32 layers. The layer-16 choice must have calibrated
   confidence, and the layer-8 verifier must agree. Other questions continue
   through the full model and use Kev's original pointer head.

Kev scores options in a forward pass, with no answer-token loop. The gain here
comes from less GPU work per decision and a kernel tuned for this model's
recurrence. Early-exit probabilities can differ from full Kev probabilities;
the [model card](models/README.md) explains the gate and quality checks.

## Performance impact

The [v0.1.0 measurement](docs/performance.md#conditional-exit-candidate) used
the same Kev-4B checkpoint on an M5 Pro, with 50 timed requests after 10 warmups
per run. The fixed request had about 275 state tokens and five three-option
questions. These are **complete HTTP request medians**, including encoding and
response formatting:

| Request | Stock Kev MLX | Packed kernel only | Packed kernel + early exit |
|---|---:|---:|---:|
| New state | 194–197 ms | 187 ms | 95 ms |
| Cached state | 80–81 ms | 76 ms | 40 ms |

The packed kernel alone improved this complete request by about **6%** without
new weights. The verified exit improved its median by about **2×**. The exit
rate depends on the question: a varied development set improved only **1.09×**
in median model time, and the candidate's new-state HTTP p95 was about 175 ms.
These fixed-request medians are not a promise for every workload. See the
[versioned evidence](evidence/conditional-exit-e23ddd3.json) for hashes, raw
samples, p95 values, and measurement order.

The frozen locked evaluation had **zero choice flips across 2,204 questions**
and full coverage. Accuracy matched stock Kev; Brier improved slightly; ECE
rose slightly within the predeclared limit. The [quality table and paired
intervals](docs/performance.md#conditional-exit-candidate) show the full result.

## Run it

On Apple Silicon with Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra serve --group dev
mkdir -p runs/weights
gh release download v0.1.0 --repo cgasgarth/nerqova --pattern 'kev4b-*' --dir runs/weights
uv run python -m nerqova.serve --run jaredpalmer/kev-4b --packed-delta \
  --exit-head runs/weights/kev4b-exit16-pair.pt --exit-threshold 0.90 \
  --verify-head runs/weights/kev4b-verify8-pair.pt --verify-threshold 0.26 \
  --port 8009
```

Send a Kev System One `POST /v1/systemone` request to `http://127.0.0.1:8009`.
Omit both head flags to run the same-weight packed kernel alone. Omit
`--packed-delta` as well to run the plain Nerqova scorer.

## Verify and measure

```bash
uv run python -m pytest tests -q -m 'not model'
uv run python scripts/bench_complete.py --engine kev-mlx --out runs/kev-complete.json
uv run python scripts/bench_complete.py --engine nerqova-packed --out runs/packed-complete.json
uv run python scripts/compare_engines.py --complete runs/kev-complete.json runs/packed-complete.json
```

Run stock and candidate measurements in separate quiet windows on the same
Mac. The [performance report](docs/performance.md) separates model-only, local
complete-decision, and HTTP time. It also records rejected kernel probes and
the limits of the measured gains.

## Source and license

`src/nerqova/` contains the checkpoint loader, scorer, packed Metal kernel, and
server. `models/` documents the trained heads. `evals/` holds suite manifests;
`evidence/` holds versioned reports. The Kev source is pinned to
`557598fced1dada75dfbf36ed144dce309ac6ceb`; [NOTICE](NOTICE) records its
attribution.

Nerqova's original code is Apache-2.0. The packed kernel adapts MIT-licensed
MLX-LM code. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the
[MLX-LM license](third_party/mlx_lm_LICENSE).

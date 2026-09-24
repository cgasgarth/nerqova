# Nerqova

Nerqova is a model-specific MLX runtime for typed System One decisions on Apple
Silicon. It reads one state and one or more questions, then returns a probability
distribution for each answer. It does not generate text. The default path uses a
packed Metal recurrence and the unchanged, pinned 4B reference weights.

## Run

On Apple Silicon, install [uv](https://docs.astral.sh/uv/) and run:

```bash
git clone https://github.com/cgasgarth/nerqova.git
cd nerqova
uv sync --extra serve --no-dev --frozen
uv run --no-dev python -m nerqova.serve \
  --run jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778 \
  --packed-delta --port 8009
```

The first start downloads the pinned public checkpoint and its base model. Send
one request to `POST http://127.0.0.1:8009/v1/systemone`:

```bash
curl -sS http://127.0.0.1:8009/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"model":"nerqova","state":"Two blue tiles can merge when moved up.","questions":{"move":{"type":"choice","instructions":"Choose the move that merges them.","criteria":{"up":"Move up","left":"Move left","right":"Move right"}}}}'
```

The response includes a choice and calibrated probabilities. The default path
needs no extra head files. The optional trained exit head and its quality limits
are in [the model card](models/README.md).

## How it works

- The model-specific Metal DeltaNet kernel packs eight value rows into each
  SIMD group. It uses the same backbone, adapter, and full pointer head as the
  reference checkpoint.
- The scorer keeps the state prefix in MLX for repeated-state requests. Its
  one-question path can capture the prefix while it scores the question.
- Loading merges adapter weights one tensor at a time. The server bounds
  reusable MLX buffers to 1 GiB. These choices reduce ready-process memory.
- An optional layer-16 readout can stop on a clear winner. Its gate compares
  the top two answer probabilities, with a separate calibrated rule for 14 or
  more options. The default server does not enable it because it did not
  improve the fixed complete-request workload.

## Measured impact

Complete local decisions on an Apple M5 Pro, macOS 26.5.2, MLX 0.32.2, and
mlx-lm 0.31.3. Each row used 5 warmups and 30 timed requests in a separate
process. “Cold” clears the state-prefix cache; “cached” repeats the state.
Both engines used the same pinned checkpoint and returned the same choice on
these fixtures. macOS reported Low Power Mode despite a selected High Power UI
setting and a 94 W adapter; the reports record that conflict.

| Request | Reference MLX cold / cached | Nerqova packed cold / cached | Speedup cold / cached |
| --- | ---: | ---: | ---: |
| One question, 3 options | 403.50 / 140.41 ms | 260.45 / 133.54 ms | 1.55× / 1.05× |
| One question, 15 options | 206.29 / 84.13 ms | 149.67 / 79.87 ms | 1.38× / 1.05× |
| Five questions, 3 options each | 202.55 / 82.51 ms | 197.18 / 80.47 ms | 1.03× / 1.03× |

The five-question request is about **5.07 decisions/s cold** and **12.43
decisions/s cached** on the packed path. These are serial complete-request
rates, not GPU-only throughput. The one-question and five-question paths have
different cache and branch work, so option and question count matter. The
2× cold and cached target was **not met** on these workloads.

On the same five-question fixture, 4-bit quantization of the reference model
reduced active MLX memory from **7.90 to 2.27 GiB**, but ran at **204.60 /
85.91 ms** and reduced transfer accuracy. A separate 0.8B reference ran at
**43.94 / 19.39 ms**, with lower transfer accuracy. The [comparison report](docs/latest-performance.md)
shows accuracy, calibration, memory, p95 latency, and each optimization's
contribution. The [versioned evidence](evidence/r8-v0.2.json) records report
hashes and raw sample summaries.

## Quality

On frozen development questions, the packed path changed **zero top choices**
against the same-weight reference across decision-v7 and transfer-v4 (2,232
questions). The largest probability difference was 0.00035. The pinned
reference reached **87.34%** decision-v7 clean accuracy, **80.49%** transfer-v4
clean accuracy, and **89.35%** documents-v1 clean accuracy with full coverage.
The optional trained exit preserved these development choices under its selected
gate, but its probabilities can differ more. [See the measured limits](docs/latest-performance.md).
On the locked decision-v7 and transfer-v4 tests, the default packed path again
changed **zero choices across 2,204 questions** and matched stock accuracy.

## Verify

```bash
uv sync --extra serve --group dev --frozen
uv run python -m pytest tests -q -m 'not model'
NERQOVA_TEST_RUN=jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778 \
  uv run python -m pytest tests -q -m model
uv run python scripts/bench_complete.py --engine kev-mlx --out runs/reference.json
uv run python scripts/bench_complete.py --engine nerqova-packed --out runs/nerqova.json
uv run python scripts/compare_engines.py --complete runs/reference.json runs/nerqova.json
```

Run paired benchmarks in quiet windows on the same Mac. `runs/` is ignored.
The separate [System One Computer Use](https://github.com/cgasgarth/system-one-computer-use)
project connects typed text or Handy dictation, Cua Driver observations, and
any System One decision endpoint to browser and desktop actions.

## Provenance and license

Nerqova's source is Apache-2.0. Its packed kernel adapts MIT-licensed MLX-LM
code. The public 4B checkpoint and the pinned Kev source dependency are research
inputs, not weights created by this repository. See [NOTICE](NOTICE),
[the model card](models/README.md), and [the performance record](docs/latest-performance.md)
for revisions, hashes, training, and quality limits.

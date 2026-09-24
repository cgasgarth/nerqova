# Nerqova conditional exit heads

These are Nerqova's trained decision readouts for the released Kev-4B backbone. The backbone, LoRA adapter, and full Kev pointer head remain unchanged. The layer-16 head can answer a request early; the layer-8 head checks agreement. A question continues through all 32 layers if the layer-16 confidence is below **0.90**, the layer-8 confidence margin above uniform chance is below **0.26**, or their top choices differ. The margin is `(top_probability - 1 / option_count) / (1 - 1 / option_count)`. The full path retains the original Kev readout.

| Release asset | Layer | SHA-256 |
| --- | ---: | --- |
| `kev4b-exit16-pair.pt` | 16 | `18ee6b7db1934325aa481199ea7b867f14b26ba02d234aee4c4f80bfe5180d11` |
| `kev4b-verify8-pair.pt` | 8 | `5c24356fba369816250226941eb781916393a15853235d6bac60b0f2f221d066` |
| `kev4b-v7-train-teacher.json.gz` | training targets | `345d4d485d5db146187cc12cfbfefd4c9b1ff852baec9bb1e61d26d0c7b2b39b` |

Both heads use a 384-wide nonlinear pair readout over the normalized decision and option anchors. They were trained on **12,576** frozen decision-v7 training records (**15,576** questions), with hard labels and the released Kev-4B probability targets at equal loss weight. The separate decision-v7 calibration split has **968** records (**1,148** questions). Training stopped at the lowest calibration NLL and fitted one temperature per head. The training manifest SHA-256 is `a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2`; the **uncompressed** teacher target SHA-256 is `eee4a3a02b19f9827e9475d2347ae8671b4ea9c0c0e2d4d1b0aff7bb52930d9b`.

The heads require Kev-4B snapshot `485ace8703592fcf405488b262449990824cfed1` and Qwen3.5-4B base revision `1001bb4d826a52d1f399e183466143f4da7b741b`. Nerqova checks the checkpoint revision when it loads them. The three assets are outside Git in the [v0.1.0 release](https://github.com/cgasgarth/nerqova/releases/tag/v0.1.0); download them to ignored `runs/weights/`. `scripts/early_exit_features.py`, `scripts/train_early_exit.py`, and `scripts/select_exit_gate.py` reproduce the feature, training, and calibration steps. Extracted feature shards stay under ignored `runs/`.

```bash
mkdir -p runs/weights
gh release download v0.1.0 --repo cgasgarth/nerqova --pattern 'kev4b-*' --dir runs/weights
```

```bash
uv run python scripts/early_exit_features.py --split train --layer 16 --out runs/early-exit/train-l16
uv run python scripts/early_exit_features.py --split calibration --layer 16 --out runs/early-exit/calibration-l16
uv run python scripts/train_early_exit.py --train-dir runs/early-exit/train-l16 \
  --calibration-dir runs/early-exit/calibration-l16 --out runs/early-exit/head-l16-pair.pt
# Repeat the three commands with layer 8 and its own directories.
uv run python scripts/select_exit_gate.py runs/early-exit/head-l16-pair.pt \
  --verify-head runs/early-exit/head-l8-pair.pt --verify-threshold 0.26
```

```bash
uv run --no-dev python -m nerqova.serve \
  --run jaredpalmer/kev-4b@485ace8703592fcf405488b262449990824cfed1 \
  --packed-delta \
  --exit-head runs/weights/kev4b-exit16-pair.pt --exit-threshold 0.90 \
  --verify-head runs/weights/kev4b-verify8-pair.pt --verify-threshold 0.26
```

The frozen development suites kept full coverage and matched stock accuracy. On decision-v7 clean questions, stock / conditional Brier was **0.18478 / 0.18423**, ECE **0.02318 / 0.02413**, and NLL **0.35832 / 0.35335**. On transfer-v4 clean questions, Brier was **0.26436 / 0.26396**, ECE **0.03635 / 0.04227**, and NLL **0.47094 / 0.46791**. ECE rose slightly on both suites; paired uncertainty intervals include zero. The varied decision-v7 median model time improved only about **1.09×** because most records deferred. See [performance evidence](../docs/performance.md) for the complete request benchmark and its limits.

The one-time locked comparison also had full coverage and zero choice flips across 2,204 questions. Decision-v7 accuracy was **0.87167** for both engines; transfer-v4 accuracy was **0.83384** for both. Brier improved from **0.18895 to 0.18837** and from **0.23205 to 0.23155**. ECE rose from **0.01924 to 0.01993** and from **0.03556 to 0.04332**. Both ECE differences met the predeclared 0.010 limit, and paired 95% intervals included zero. The [versioned evidence](../evidence/conditional-exit-e23ddd3.json) records the exact hashes and samples.

These heads were trained for Kev's typed decisions. They have no computer-use or browser-action quality claim.

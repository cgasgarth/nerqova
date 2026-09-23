# Same-weight performance plan

The primary target is the released Kev-4B checkpoint on an Apple M5 Pro with 20 GPU cores and 48 GB of memory. Compare Nerqova with stock Kev MLX using the same adapter, base revision, head, temperature, encoded request, and warm resident model. Keep the returned choices fixed and measure probability differences. A new trained model is separate research; a smaller model alone does not satisfy this target.

Kev scores all options in one pass. It does not write a sequence of answer tokens. [Husky's](https://husky.underdog.ai/) draft-token and block-verification method does not apply. Its model-specific Metal kernels, packed weight layout, and GPU scheduling are the relevant hardware ideas. Test each on Kev-4B's prefill and branch workload; do not infer a gain from Husky's token-generation measurements.

## Measured starting point

On one quiet M5 Pro run at code revision `7ca9122`, with 50 repetitions after 10 warmups, a roughly 275-token state and five three-option questions:

| Boundary, median ms | Stock Kev MLX | Nerqova |
|---|---:|---:|
| Pre-encoded model, new state | 192.75 | 193.96 |
| Pre-encoded model, cached state | 77.46 | 78.35 |
| Complete local decision, new state | 197.63 | 196.30 |
| Complete local decision, cached state | 81.29 | 79.83 |

The input and checkpoint hashes matched. All five choices matched; maximum probability difference was 0.00022. The two engines are effectively tied at this stage. These are one quiet block per engine, so they establish a starting point rather than a release speed claim. HTTP timing remains to be measured.

Earlier barrier-instrumented profiling of the cached Kev-4B branch attributed about 45 ms to its 32 MLPs, 37 ms to its 24 Gated DeltaNet layers, and 9.5 ms to its eight full-attention layers. Barriers add time, so those parts do not sum to the uninstrumented request. The first custom kernel must target a large measured part and preserve model output.

## Hardware work

1. Save an exact stock MLX baseline on several request shapes. Profile GPU kernel time and host gaps without using the trace run as the timing result.
2. Build one model-specific Metal change for the Qwen3.5-4B shapes. Start with the kernel family that dominates the quiet run. Repack weights or recurrent state only if the kernel reads the new layout directly.
3. Compare every change against stock MLX on identical requests. Check the pristine state cache, independent question branches, option order, every choice, maximum probability difference, and frozen development-suite metrics.
4. Measure pre-encoded model time, complete local decision time, and HTTP latency separately. Alternate baseline and candidate runs on a quiet GPU. Record pinned checkpoint and code hashes, hardware, runtime versions, median, and p95.

The first goal is a **measured same-weight speedup**. The stretch target is at least 2× lower complete decision latency. Do not claim that target until the full path passes the correctness gate and repeated timing runs.

## Packed Metal result

The optional packed Gated DeltaNet kernel assigns eight value rows to one SIMD group. It changes the recurrence layout without changing checkpoint weights. On the M5 Pro, its output and final state were bitwise equal to the prior recurrence kernel on the tested 4B shapes. Across 2,232 frozen development questions, the packed scorer returned exactly the same probabilities as Nerqova's prior scorer. Against stock Kev MLX there were no choice flips; maximum probability difference was 0.00040 on decision-v7 and 0.00019 on transfer-v4. All 764 transfer questions were scored. The locked test stayed closed.

At clean commit `601727c`, with 50 repetitions after 10 warmups in a quiet window:

| Complete decision median | Stock Kev MLX | Packed Nerqova | Speedup |
|---|---:|---:|---:|
| Local, new state | 196.03 ms | 184.75 ms | 1.061× |
| Local, cached state | 79.73 ms | 75.00 ms | 1.063× |
| HTTP, new state | 197.53 ms | 186.51 ms | 1.059× |
| HTTP, cached state | 80.78 ms | 76.21 ms | 1.060× |

The full [evidence summary](../evidence/packed-metal-601727c.json) records the source, checkpoint, suite hashes, served metrics, coverage, p95 values, and input hashes. An earlier reverse-order local block also showed a gain. This is a real model-specific hardware improvement. It does not meet the 2× target.

## Conditional exit candidate

The two [trained heads](../models/README.md) use the same packed Kev-4B backbone. A layer-16 readout must reach 0.90 confidence; a layer-8 readout must choose the same option and reach a **0.26 confidence margin above uniform chance**. The margin is `(top_probability - 1 / option_count) / (1 - 1 / option_count)`, so questions with different option counts use the same scale. Other questions continue through all 32 layers and use Kev's original pointer head. A mixed-batch fallback test matches the full packed path within `1e-4` probability. This is a model change: early probabilities can differ substantially from full Kev, so quality is checked on held-out records.

On the frozen decision-v7 calibration partition, the gate exited 31.0% of questions and 21.6% of complete requests. It matched full Kev accuracy, had no teacher choice flips among accepted exits, and slightly improved Brier, ECE, and NLL. Neither head saw calibration records during training.

An **exploratory** fixed-request local block and a separate HTTP block used the same 275-token/five-question request. The HTTP new-state block varied the state text to force cache misses:

| Median complete decision | Stock Kev MLX | Conditional Nerqova | Speedup |
|---|---:|---:|---:|
| Local, new state | 194.75 ms | 91.91 ms | 2.119× |
| Local, cached state | 78.92 ms | 37.70 ms | 2.093× |
| HTTP, new state | 196.13 ms | 94.84 ms | 2.068× |
| HTTP, cached state | 80.21 ms | 39.55 ms | 2.028× |

All five final choices matched. These exploratory runs cross code revisions and include a dirty checkout, so they are not a release speed claim. Three of 50 cold HTTP requests deferred; the candidate's cold p95 was **175.48 ms**. A clean-commit 50-repetition local and HTTP comparison is required.

On frozen development records, stock Kev MLX and the conditional scorer had **zero choice flips** across 1,468 decision-v7 and 764 transfer-v4 questions, with full coverage. Clean-question accuracy was unchanged: 0.87184 on decision-v7 and 0.80030 on transfer-v4. Brier improved from 0.18478 to 0.18423 and from 0.26436 to 0.26396; NLL also improved. ECE rose from 0.02318 to 0.02413 and from 0.03635 to 0.04227. A 1,000-resample paired record-cluster bootstrap gave ECE-difference 95% intervals of `[-0.00206, 0.00334]` and `[-0.00172, 0.00837]`; both include zero. The locked test remains closed at this stage.

The verifier is conservative. It fully exited 375/1,204 decision-v7 development requests and 149/764 transfer-v4 requests. On the varied decision-v7 set, median model time improved only from 111.56 to 102.08 ms (**1.09×**). The fixed-request speedup must not be generalized to every request. The exit and verifier selection used calibration; development results remain an exploratory check until the final locked test.

Freeze the checkpoint, both head artifacts, and gate before the one locked-test read. The confirmatory quality limits are full coverage, no clean accuracy loss, at most **0.005** higher Brier and **0.010** higher ECE on each suite, with the paired ECE interval including zero. The fixed-request speed target is a **2.0×** or greater median improvement for both new and cached states, locally and through HTTP, in 50-repetition clean-code runs. A failed locked result must be reported, not tuned against that partition.

## Rejected probes

The following short M5 Pro probes used the same Kev-4B weights. They are diagnostic, not release benchmarks. Their code paths were removed after the full request failed the gate.

| Change | Isolated result | Full fixed request |
|---|---|---|
| Fused 2560-wide residual add and RMSNorm Metal kernel | 1.35× faster on a 275-token tensor; slower on the short branch | About 1.00× stock MLX; maximum probability difference 0.0035 |
| DeltaNet Metal threadgroup rows 4 → 8, 16, 32 | No improvement over MLX-LM's 4-row launch | No full-model run after the isolated loss |
| Four DeltaNet input projections packed into one matrix | 1.06–1.08× faster on one layer | About 0.93× stock MLX; maximum probability difference 0.0066 |
| Metal 4 fused gate/up MLP probe, 64×64 and 64×128 tiles | 1.03–1.04× faster on one projection pair | No full-model run; the isolated gain did not justify replacing MLX's tested GEMMs |
| M5 Pro NAX GEMM tile widened from 64×128 to 128×128 for large projections | Mixed isolated results | 0.95× new-state and 0.90× cached-state speed relative to the unmodified source build; rejected |

These results show why a kernel-only improvement cannot establish a decision-speed gain. Use bounded CLI measurements to select the next full-request bottleneck. A three-request Xcode GPU capture expanded to 19 GB on disk and 124 GB during replay; that method is excluded from further work on this Mac.

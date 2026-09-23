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

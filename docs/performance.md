# Same-weight performance plan

The primary target is the released Kev-4B checkpoint on an Apple M5 Pro with 20 GPU cores and 48 GB of memory. Compare Nerqova with stock Kev MLX using the same adapter, base revision, head, temperature, encoded request, and warm resident model. Keep the returned choices fixed and measure probability differences. A new trained model is separate research; a smaller model alone does not satisfy this target.

Kev scores all options in one pass. It does not write a sequence of answer tokens. [Husky's](https://husky.underdog.ai/) draft-token and block-verification method does not apply. Its model-specific Metal kernels, packed weight layout, and GPU scheduling are the relevant hardware ideas. Test each on Kev-4B's prefill and branch workload; do not infer a gain from Husky's token-generation measurements.

## Measured starting point

On one quiet M5 Pro smoke run with a roughly 275-token state and five three-option questions, stock Kev MLX took **191 ms** for a new state and **77 ms** with a cached state. Nerqova's current MLX backbone and native pointer head took **189 ms** and **75 ms**. The encoded input and checkpoint matched; all five choices matched; maximum probability difference was **0.00022**. These ten-repetition smoke numbers locate the starting point. They are not a release benchmark.

Earlier barrier-instrumented profiling of the cached Kev-4B branch attributed about 45 ms to its 32 MLPs, 37 ms to its 24 Gated DeltaNet layers, and 9.5 ms to its eight full-attention layers. Barriers add time, so those parts do not sum to the uninstrumented request. The first custom kernel must target a large measured part and preserve model output.

## Hardware work

1. Save an exact stock MLX baseline on several request shapes. Profile GPU kernel time and host gaps without using the trace run as the timing result.
2. Build one model-specific Metal change for the Qwen3.5-4B shapes. Start with the kernel family that dominates the quiet run. Repack weights or recurrent state only if the kernel reads the new layout directly.
3. Compare every change against stock MLX on identical requests. Check the pristine state cache, independent question branches, option order, every choice, maximum probability difference, and frozen development-suite metrics.
4. Measure pre-encoded model time, complete local decision time, and HTTP latency separately. Alternate baseline and candidate runs on a quiet GPU. Record pinned checkpoint and code hashes, hardware, runtime versions, median, and p95.

The first goal is a **measured same-weight speedup**. The stretch target is at least 2× lower complete decision latency. Do not claim that target until the full path passes the correctness gate and repeated timing runs.

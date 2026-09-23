# Performance plan

Nerqova targets at least 2× lower complete decision latency than the current Kev-4B MLX service on the same Apple M5 Pro request. It must also match or improve held-out decision accuracy and served probability quality.

Kev scores all options in one pass. It does not produce a long output token stream. [Husky's](https://husky.underdog.ai/) block verification and draft-token method therefore does not map to this workload. Its model-specific kernels, packed weight layout, and GPU scheduling are relevant ideas, but each needs a measured gain on Nerqova's own shapes.

## Current path

1. Train a smaller Qwen3.5-2B pointer model on Kev's frozen training partition.
2. Serve it through the Nerqova MLX scorer. Keep the state prefix cache and the question branches separate.
3. Measure new-state and cached-state model time, complete local decision time, and HTTP time on the same requests as Kev-4B.
4. Profile the trained model. Change kernels or weight layout only for the stages that dominate its time. Check probabilities and option choices after each runtime change.

An early architecture probe on the M5 Pro measured about 75 ms for a new-state 2B request and 208 ms for Kev-4B MLX. The 2B probe did not establish model quality, and these times are not release results. The fixed request had about 275 state tokens, 430 packed tokens, and five three-option questions. The release benchmark will use a quiet GPU, a committed code revision, a pinned checkpoint, repeated runs, and the same request for both models.

The [development baseline report](../evidence/kev4b-transfer-v4-development.json) defines the initial quality comparison. A finalist also needs a new paired run with pinned checkpoint revisions and a locked-test read.

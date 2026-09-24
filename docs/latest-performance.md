# Pinned 4B runtime: quality and speed

This record compares the same public 4B decision checkpoint on one Mac. The
checkpoint is `jaredpalmer/kev-4b@1da696f7938f77c4cdf5471e92fd342baff41778`,
with base revision `1001bb4d826a52d1f399e183466143f4da7b741b`, adapter
SHA-256 `c2c99077b58cd4a604304b743b7db5391187ef1d58e575f90c554c744aa23d65`,
and full head SHA-256
`9a3051c3676f0be29d885935b6f50c287f7acf04e3cb29a6fed82ca3e5540b7e`.
The reference is stock Kev MLX with this same checkpoint, not PyTorch or an
older weight file. The upstream [model card](https://huggingface.co/jaredpalmer/kev-4b)
describes the checkpoint's document fine-tune and its source limits.

## Development quality

All rows use the same frozen items and served temperatures. Accuracy, Brier,
and ECE are for clean questions. Higher accuracy is better; lower Brier/ECE is
better. Coverage was 100% for every row. No locked test data selected the
runtime or gate.

| Path | decision-v7 accuracy / Brier / ECE | transfer-v4 accuracy / Brier / ECE |
| --- | --- | --- |
| Reference 4B MLX | .87342 / .18137 / .02492 | .80488 / .26574 / .04152 |
| Nerqova packed | .87342 / .18138 / .02491 | .80488 / .26573 / .04146 |
| Nerqova conditional exit | .87342 / .18071 / .02644 | .80488 / .26568 / .04268 |
| Reference 4B, 4-bit weights | .87500 / .18576 / .02498 | .78506 / .28282 / .06205 |
| Reference 0.8B MLX | .82832 / .24834 / .02613 | .65091 / .43076 / .05862 |

The packed path changed zero top choices across all 1,468 decision-v7 and 764
transfer-v4 development questions. Its maximum probability difference was
0.00035. The conditional exit also changed zero choices on those questions;
its probabilities can differ by as much as 0.399 on a question. Its selected
log-odds gate was 5.0 for 2–13 options and 2.5 for 14 or more options. It
exited on 423 / 1,468 decision questions and 95 / 764 transfer questions;
complete-record exits were 364 / 1,204 and 95 / 764.

On `documents-v1` development, the reference scored .89348 accuracy, .16087
Brier, and .05385 ECE across 920 questions in 568 CFPB complaint records. The
conditional exit matched accuracy, changed zero top choices, and scored .16087
Brier / .05384 ECE. It exited on only 45 questions and 11 complete records;
states beyond the head's 384-token training context always use the full model.
This documents result is about the CFPB source and question templates in the
upstream suite. It does not establish quality on unrelated documents.

### One locked comparison of the default path

After code freeze at `d93a692cf8e168c6ab7cd0cd6557d905ca9767e6`, stock
MLX and the default packed path each read the decision-v7 and transfer-v4
locked partitions once. Both covered every record and matched accuracy:

| Locked partition | Stock accuracy / Brier / ECE | Packed accuracy / Brier / ECE | Top-choice changes |
| --- | --- | --- | ---: |
| decision-v7, 1,440 questions | .87500 / .181992 / .024896 | .87500 / .181997 / .024947 | 0 |
| transfer-v4, 764 questions | .83384 / .233569 / .036057 | .83384 / .233564 / .036048 | 0 |

The maximum probability difference was 0.00036 on decision-v7 and 0.00019 on
transfer-v4. The optional exit head failed its speed gate and **was not** read
on the locked partitions. The [versioned evidence](../evidence/r8-v0.2.json)
contains run hashes, coverage, paired row parity, and the complete-request
sample summaries.

## Complete-request latency

Apple M5 Pro, macOS 26.5.2, MLX 0.32.2, mlx-lm 0.31.3, Python 3.13.15.
Every engine ran in its own process with a 1 GiB MLX reusable-buffer cap.
Each condition used five warmups and 30 timed complete in-process requests;
encoding and response formatting are included. “Cold” clears the state-prefix
cache each time; “cached” repeats the state. The 94 W Apple adapter was
connected. System Settings showed High Power, while `pmset` and
`system_profiler` reported Low Power Mode. All paired paths were measured under
that reported state. These timings are not an HTTP, browser, or full agent task
benchmark.

| Fixture and path | Cold median / p95 ms | Cached median / p95 ms | Active MLX GiB |
| --- | ---: | ---: | ---: |
| Five questions, reference 4B | 202.55 / 203.50 | 82.51 / 83.37 | 7.90 |
| Five questions, Nerqova plain | 206.59 / 207.67 | 83.58 / 84.75 | 7.90 |
| Five questions, Nerqova packed | 197.18 / 198.62 | 80.47 / 81.36 | 7.90 |
| Five questions, conditional exit | 198.18 / 200.44 | 81.47 / 82.64 | 7.91 |
| Five questions, reference 4B 4-bit | 204.60 / 205.26 | 85.91 / 86.37 | 2.27 |
| Five questions, reference 0.8B | 43.94 / 45.12 | 19.39 / 19.74 | 1.43 |
| One question, 3 options, reference 4B | 403.50 / 562.99 | 140.41 / 177.98 | 7.90 |
| One question, 3 options, packed | 260.45 / 384.94 | 133.54 / 169.99 | 7.90 |
| One question, 15 options, reference 4B | 206.29 / 207.71 | 84.13 / 85.07 | 7.90 |
| One question, 15 options, packed | 149.67 / 150.18 | 79.87 / 80.72 | 7.90 |
| One question, 15 options, conditional exit | 197.50 / 206.40 | 81.66 / 82.06 | 7.91 |

The 15-option fixture used repeated, numbered action descriptions to measure
its request shape; it is not a computer-use accuracy test. One-question and
five-question timings differ because their state and branch work use different
paths. The one-question three-option p95 was more variable than the other
fixtures. The default packed path improved cold latency by **1.55×** on that
fixture, **1.38×** on one question with 15 options, and **1.03×** on five
questions; cached gains were **1.05×**, **1.05×**, and **1.03×**. The target of
at least **2× cold and cached** was not met.

## Contribution and alternatives

For the 15-option request, plain Nerqova was 207.69 / 84.63 ms and the unmasked
branch path without the packed kernel was 207.89 / 85.29 ms. The packed path was
149.67 / 79.87 ms. This measures the combined packed-kernel and one-pass
branch path; the unmasked branch change alone did not help. On the five-question
request, the packed path was only about 5% faster than plain Nerqova and 3%
faster than the stock reference. The new conditional exit did not improve either
fixed workload despite passing development quality gates.

The 4-bit reference cut steady active MLX memory from 7.90 to 2.27 GiB. It was
slower than bf16 and lost 1.98 percentage points of transfer accuracy. The
conversion happens after the bf16 checkpoint is loaded, so process peak RSS
remained high. The 0.8B reference was about 4.5× faster cold and 4.2× faster
cached than packed 4B on the five-question fixture, but lost 15.4 percentage
points of transfer accuracy. These paths have different speed, quality, and
memory tradeoffs; a single accuracy-times-speed number would hide that loss.

### Complete loopback HTTP requests

The same pinned 4B checkpoint was served through stock Kev MLX and Nerqova's
packed server in separate quiet processes, each with a 1 GiB MLX reuse cap.
Each report has five warmups and 30 measured requests. Cold requests used
distinct state strings; cached requests repeated one state. Requests and final
choices matched within each pair. HTTP transport, encoding, model work, and
response formatting are included.

| Request | Stock cold / cached median ms | Packed cold / cached median ms | Speedup cold / cached |
| --- | ---: | ---: | ---: |
| Five questions, 3 options each | 208.26 / 85.48 | 194.32 / 79.56 | 1.07× / 1.07× |
| One question, 15 options | 210.07 / 86.55 | 154.56 / 81.91 | 1.36× / 1.06× |

The [HTTP evidence](../evidence/http-r8-v0.2.json) records request hashes,
report hashes, choices, and p95 values. These calls exclude browser and desktop
observation time. The 2× cold/cached target was not met on either fixture.

Two further model-specific candidates were measured and left out of the
runtime. A terminal DeltaNet branch omitted final recurrent-state outputs and
passed 28 parity checks. Its alternating complete-request blocks did not
repeat a 5% gain: candidate medians were 190.0–190.8 / 75.9–76.8 ms and
baseline medians 191.0–199.8 / 77.5–79.4 ms, with drift in the last baseline.
A fused gate/up Metal MLP prototype took about 13.9 ms per tested layer block,
versus 1.3 ms for MLX, and was not bitwise equal. Both prototypes remain under
ignored `runs/` for research. Neither supports a speed claim.

The old [v0.1 performance report](performance.md) applies to a different 4B
checkpoint and head files. Do not combine its speed or quality numbers with
this version.

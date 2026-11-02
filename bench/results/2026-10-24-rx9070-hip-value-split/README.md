# RX 9070 XT: opt-in HIP split value reduction

Measured on 2026-10-24/25 by leakedd against PolyStrata
`c174f35c2a5e20a4fe7ac236063d6513fcc07179`. This report accompanies the
attention change; it is not a general GPU or model-quality comparison.

## What changes

`STRATA_HIP_VALUE_SPLIT=1` enables a two-stage weighted-value reduction for
head dimension 256 / GQA8 at 12,288 or more keys. Keys are divided into 1,024-key parts; each part
reuses values across eight query heads, then a second kernel joins the parts.
It uses the dead score buffer as scratch and leaves transposed softmax weights
intact. FP16 and Q8_0 KV paths are covered. It is disabled by default and falls
back for unsupported geometry, windows/rings, whole attention, short contexts,
or insufficient scratch. FP32 addition order changes when enabled.

PolyStrata's merged PR #51 already splits CUDA decode attention into 1,024-key
parts with tensor cores and states that HIP keeps the old path. This report
concerns the HIP reduction gap, not a claim of inventing sequence-split
attention. The broader method is established by Flash-Decoding and the
FlashAttention work.

## Applicable models

The kernel targets the D256/GQA8 full-attention path used by Qwen3.6-35B-A3B
MoE. Its published configuration has head dimension 256, 16 query heads, two
KV heads and 10 full-attention layers among 40. Variants with the same geometry
are eligible under the runtime gates above; linear-attention layers are
unaffected. End-to-end gains depend on quantization and CPU/GPU expert placement.

## Machine and build

- RX 9070 XT, 16 GiB, gfx1201; PCIe 32 GT/s x16 observed.
- Ryzen 5 9600X, six cores / twelve threads; 32 GB installed system RAM.
- Ubuntu 24.04.5 LTS; Linux 7.0.0-38-generic; amdgpu driver.
- Source Release HIP build, gfx1201, two build workers, tests enabled.
  ROCm 10.1 SDK wheel libraries, HIP 7.16, LLVM 24. Exact build identifiers and
  model/tokenizer hashes are recorded in `provenance.json`.
- llama.cpp dependency pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Isolated benchmark server; ordinary inference service stopped during timings.
  CUDA downloads/compilation did not run during measured inference. No GPU
  clock or power-limit changes were made for this comparison. Power/temperature
  telemetry and storage performance were not measured.

## Benchmark representative and method

Cyber-Tiel-Coder-35B-A3B UD-Q4_K_XL, a community `qwen35moe` derivative with
the geometry above. All performance numbers in this report were measured on
this artifact; the exact filename and hashes are in `provenance.json`.
Context allocation 163,840;
Q8_0 KV, whole KV allocation, prompt batch 2,048, reserve 1,536 MiB,
conversation cache 2,048 MiB, six CPU workers, graphs disabled. Routed expert
placement starts from one frozen profile; swaps, CPU share and tail skipping
are zero. Suffix lookup is explicitly disabled (`--lookup 0`). No vision,
calibration or throughput projection is involved.

The main performance comparison uses **one candidate binary**, changing only
`STRATA_HIP_VALUE_SPLIT=0` versus `=1`. Each arm restarts the engine and begins
from the same profile. A separate unpatched binary validates default behavior.
The main runs use `--no-draft --window 1`, and target ordinary
autoregressive decode. MTP was exercised separately during exploratory work
and is not included in the performance claims below.

[`reproduce.py`](reproduce.py) builds prompts from the pinned base commit's
public Markdown documentation and the model's tokenizer. Its `--dry-run` mode
records corpus and prompt hashes without an API request. The request uses
one user message, temperature 0, top-p 1, top-k 1, seed 42, repetition penalty 1,
and thinking disabled. A 32-token warm-up precedes each arm and is excluded.
The task requests an LRU cache, explanation and tests; output is capped at
512 tokens (768 for MTP). Those truncated outputs are speed workloads, not
completed coding-quality tests. Load time is excluded.

The requested depths are corpus-token lengths; use the actual prompt token
counts in the raw data. An independent prefix for every sample avoids reusing
the previous timing prompt. Native `prompt_ms`/`predicted_ms` separate prefill
and decode. `elapsed_s` includes the whole HTTP request; it is not TTFT.
VRAM/GTT/busy samples are taken every 0.5 s. Quality checks are excluded from
the speed summary; their prefix-cache reuse is retained in the raw record.

## Results

Three restarted runs per flag on the final binary; the median and full range
are shown. The same binary, frozen expert profile and request were used. The
8,192-key row is a fallback control: the flag is inactive below 12,288 keys,
so its spread measures ordinary run-to-run noise.

| Corpus depth | Actual prompt | Runs per flag | Decode off, median (range) | Decode on, median (range) | Gain | PP off / on, median |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 8,192 | 8,441 | 3 / 3 | 56.6 (53.1-59.1) | 53.1 (52.7-56.4) | fallback; noise | 2966.5 / 2979.1 |
| 12,288 | 12,539 | 3 / 3 | 58.5 (54.6-58.7) | 59.8 (52.7-60.9) | 2.2% | 3813.8 / 3675.9 |
| 32,768 | 33,019 | 3 / 3 | 48.5 (46.6-49.6) | 52.3 (51.3-54.2) | 7.8% | 3780.5 / 3785.9 |
| 98,304 | 98,563 | 3 / 3 | 36.5 (35.6-36.6) | 44.6 (42.5-45.0) | 22.2% | 2493.3 / 2488.8 |

The 12K ranges overlap, so treat that row as no measured regression rather
than a reliable gain. The effect becomes repeatable at 32K and is largest at
98K.

All values are tokens/s. Prompt processing stays within 3.8% between flags;
the change targets decode. Peak sampled VRAM was 14.50 GiB off and 14.50 GiB
on over the main runs; GTT stayed at 153,403,392 bytes. See `timings.csv` for
every run, output hash, actual token counts and memory peaks.

The final quality arm produced byte-identical API messages with the flag off
and on for arithmetic, JSON ordering, two code tasks and a three-marker 96K
recall. Both code answers passed their supplied tests. The strict recall
validator rejects the marker-name prefixes, while both answers contain all
three values correctly. This is a smoke check, not a general quality score.

## Reproduction

Build the HIP executable and affected checks using the normal instructions in
`docs/AMD_HIP.md`, then run:

```sh
cmake --build build-hip --target strata-poly poly_hip_value_split_test \
  poly_hip_value_split_policy_test poly_kv_q8_test -j2
ctest --test-dir build-hip --output-on-failure \
  -R '^poly_(hip_value_split(_policy)?|kv_q8)_test$'
```

Start the authenticated server using the arguments in `provenance.json`, with
your own model paths and a copy of the saved expert profile. Set the flag in
the engine environment before starting it. The client reads authentication
from an environment variable; do not put credentials in a shared report.

```sh
python bench/results/2026-10-24-rx9070-hip-value-split/reproduce.py \
  --model MODEL_NAME --tokenizer TOKENIZER_DIRECTORY --arm off_A --rep 0 \
  --out off_A.jsonl
```

Restart with an identical profile and flag `1`, use `--arm on_A --rep 0`, and
repeat with independent `--rep` values. To reproduce MTP, configure its donor
and window four in the engine and also pass `--mtp` to the client. The client
labels the run; it does not change the running engine's configuration.

## Correctness and limits

- Host policy test covers explicit opt-in, geometry, thresholds, capacity and
  overflow-safe selection.
- HIP test: 414 checks covering FP16/Q8_0, static/device positions, multi-chunk
  query batches, causal boundaries, sharper scores, sinks, fallback layouts,
  graph replay, and an independent FP64 weighted-value reference. Maximum
  attention absolute difference from the original reduction: 1.14440918e-5.
  Maximum difference from the FP64 reference: 5.67598994e-8.
- Default-off baseline/candidate attention dumps: 47,185,920 bytes exactly
  equal, SHA256 `61249d8ab81931f5c4e7358345193e5d5e78ef2583c3b79db2fb852c73d88e51`.
- Default-off full-model API messages also matched at 1K and 8K with fixed CPU
  expert placement. This compares API-visible messages, not raw token IDs.
- At 8K, 64 raw CLI token IDs are identical with the final binary's flag off
  and on: SHA256 `342fea3563172c6ea2a60c0dd5cfcdc44767f8e347dcb10b6f2d4d98c4438b80`
  (`raw-token-fallback.json`). A separately configured baseline process held
  4,597 rather than 4,602 experts on its first run; that unrelated whole-model
  output comparison was discarded.
- Focused CTest suite: 3/3 passed (`final-focused-tests.txt`). Existing sink-attention test passed in both
  builds. Existing `poly_attend_test` aborts with `attention scores: cuBLAS
  status 1` on **both** unpatched and modified HIP builds; it is not reported
  as a pass. Its batch fixture supplies a null BLAS handle; the cause has not
  been independently repaired or established here.
- The changed reduction is not bitwise equivalent when enabled. Small smoke
  checks cannot establish broad model-quality equivalence. Only this AMD
  card/model has inference measurements; graph correctness is tested but
  real-model performance uses graphs disabled. No SYCL source path changes.
- CUDA source configuration and compilation passed for sm_86
  (`cuda-build.txt`); this host has no NVIDIA card, so the CUDA executable and
  GPU test were not run.

Pilot records are retained in `exploratory-runs.jsonl`. The first pilot's marker recall
validator required bare values and rejected strings containing the marker
names despite all three values being present. The second pilot had different
GPU expert counts between binaries and suffix guessing still enabled. It is
excluded from performance conclusions and full-model default equivalence.
The controlled default comparison fixes placement; measured flag comparisons
use the same executable, frozen profile and no suffix lookup. The startup and
expert-cache lines are excerpted in `engine-log-excerpt.txt`.

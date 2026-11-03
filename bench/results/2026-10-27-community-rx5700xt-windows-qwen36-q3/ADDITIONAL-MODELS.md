# Ornith, Tiel Coder and Gemma 4 QAT on the same Windows gfx1010 build

The hardware, driver, toolchain, source commit and executable are the same as
[the original Qwen report](README.md). These three additional runs used upstream
family setup with the actual device metadata adapter and a verified reuse of
the existing engine. The helper bypasses the top-level GPU allowlist and checks
the executable/source identity; no engine/kernel source was patched. This is
an unvalidated RX 5700 XT route, not a proposed support-list change.

## Exact models and configurations

| Family | Tested artifact | Decimal GB, model + draft | Revision |
|---|---|---:|---|
| Ornith | APEX-I-COMPACT, embedded MTP | 17.438 | `4c326713c52d13fdb40983a2a09a46cca8e88739` |
| Tiel | UD-Q3_K_XL MTP | 17.235 | `bb5f3f441849fc24c2d9037c310bc995404bfa0c` |
| Gemma 4 | QAT-UD-Q4_K_XL + matching QAT Q8_0 draft | 14.711 | `7b92b5b28818151e8669af2e45e88d6086f490dd` |

The per-family `model.json` files retain repository names, exact filenames,
byte sizes and SHA256 hashes. Gemma's QAT draft has SHA256
`75f58e20c32281d1adc086c0569d53f147e8d81a7b8493cf4a73a028415f67af`;
the same-named draft from another repository is not interchangeable evidence.

| Family | Official per-request JSON | Model / config / profile / quality files |
|---|---|---|
| Ornith | [results.json](additional-models/ornith/results.json) | [folder](additional-models/ornith/) |
| Tiel | [results.json](additional-models/tiel/results.json) | [folder](additional-models/tiel/) |
| Gemma | [results.json](additional-models/gemma4/results.json) | [folder](additional-models/gemma4/) |

All three tool-sanitized official JSON files are copied byte-for-byte. Each ran
33 measured requests, three repeats, configured context 32768, `--long` absent,
no parallel clients and no thinking. See the root table for medians and the JSON
for every request, range, phase timing and generated length. Calibration/warm-up
requests are outside that measured count.

The additional configs use default FP16 KV, `--kv-whole`, automatic conversation
cache (8192 MiB cap), 16 CPU workers and family MTP. **They omit `--tail-skip`**;
the explicit `--tail-skip 6` in the original Qwen config must not be attributed
to every family. Read the recorded argument lists rather than assuming identical
model-specific defaults. Vision was not enabled or tested.

Basic and coding probes ran on the same 32K server before the official suite's
own warm-up/calibration. These are warm, stateful runs. The later 64K server
reused the evolving expert profile; profile snapshots before official and before
64K are supplied as exact byte-preserved files. They are not matched profiles
between different models. Disk/model caches were not flushed.

## Custom long-context retrieval

One request per row, three planted reference codes, output allowance 128 tokens,
zero reused input tokens. Rows below include the original Qwen probes to make
the actual workload sizes clear. These are not repeated official-suite medians.

| Model / configured window | Input tokens | Output tokens | Engine prefill s | Prompt tok/s | Decode tok/s | Client total s |
|---|---:|---:|---:|---:|---:|---:|
| Qwen / 65536 | 58117 | 48 | 168.249 | 345.4 | 28.1 | 170.028 |
| Qwen / 131072 | 116115 | 48 | 498.695 | 232.8 | 22.8 | 500.899 |
| Ornith / 65536 | 58117 | 53 | 188.185 | 308.8 | 19.2 | 191.016 |
| Tiel / 65536 | 58122 | 39 | 170.647 | 340.6 | 21.5 | 172.549 |
| Gemma QAT / 65536 | 59539 | 52 | 187.151 | 318.1 | 27.9 | 190.947 |
| Gemma QAT / 131072 | 119038 | 52 | 529.203 | 224.9 | 20.1 | 536.089 |

Prefill is engine-reported prompt time; total and TTFT are client wall timings.
They have different boundaries and must not be equated. Full timing/usage fields
are in the context JSON files. These short replies provide limited decode-rate
evidence. Configured context is larger than the actual input; no full-window or
broad long-context capability is established.

Gemma's separate 128K extension used the same QAT weights/draft and FP16 KV,
a cloned post-core profile and normal WDDM budget, without a budget override or
quantized-KV fallback. It is not a controlled context-only performance curve.
Windows budget/cache layout differed between launches (roughly 7383 MiB at32K
and 6373 MiB for Gemma128), despite 8176 MiB physical VRAM. The context64/128
configs and profile snapshots are retained alongside the results.

## Correctness failures, not just successful requests

| Model | Two short probes | Coding, primary strict result | Long retrieval |
|---|---|---|---|
| Ornith | 2/2 pass | JSON accepted; 8/8 functional cases pass | All 3 codes; strict JSON fails because of Markdown fences |
| Tiel | 2/2 pass | JSON format fails; secondary offline normalization leaves unchanged code at6/8 cases | All3 codes and strict JSON pass |
| Gemma QAT | 1/2 pass; sorting is fenced | JSON format fails; secondary offline fence removal leaves unchanged code at8/8 | All3 codes at64K/128K; strict JSON fails at both |

The synthetic coding task requests a pure `merge_intervals` function, correct
touching-endpoint behavior, preservation of input and JSON-only output. Tiel's
unchanged function incorrectly merges one-unit gaps. Secondary processing is
explicitly labeled in each `coding.json`; it does not upgrade the original pass.
No model retries or regenerations were performed to fix these outcomes.
Exact coding prompts/responses, eight test cases and grades are retained. The
[original bounded grader](grade-merge-intervals.py) accepts a response object
containing `text`, uses a restricted AST/builtin allowlist, and does not strip
fences automatically. The fixture is tiny, not a coding benchmark leaderboard.

## Sampled memory

These rows cover each model's entire 32K +64K sequence, not isolated per-context
peaks. RAM was sampled at a target2seconds; GPU counters at5seconds. Peak values
can miss short spikes; physical memory, RSS, private commit and GPU shared pages
overlap and must not be added.

| Model | Whole-system peak GiB | Engine RSS peak GiB | Engine private commit peak GiB |
|---|---:|---:|---:|
| Ornith | 38.675 | 20.319 | 4.416 |
| Tiel | 38.946 | 20.516 | 5.050 |
| Gemma QAT | 39.935 | 21.423 | 7.831 |

Aggregate byte counts, baselines, sample counts and GPU dedicated/shared peaks
are in each `memory-summary.json`. Gemma128 separately peaked at35.849GiB
whole-system RAM,17.691GiB engine RSS and4.068GiB private commit. Earlier Qwen
32K/64K RAM peaks remain unknown. Cache capacity is not allocation or usage.
Whole-system values include background applications; no OS paging policy was changed.

## Reproduce

Use the pinned source/runtime from the root README, and predownload the exact
model and draft files listed in the per-family metadata. The generalized
`setup-gfx1010.py --family ornith|tiel|gemma4 --reuse-tested-engine` supplies the
measured device and verifies the retained executable/source before invoking
upstream family setup. The original top-level installer still rejects this GPU.
Recorded configs contain `<workspace>` placeholders and must be mapped to local
paths; they are evidence templates, not portable launchers. Place the verified
Gemma QAT draft where its config points, checking the SHA256 above.

```powershell
python reproduce-additional.py --verify-only
# After starting the recorded model at the selected context:
python reproduce-additional.py --source <checkout> --model ornith --context 65536
python reproduce-additional.py --source <checkout> --model gemma4 --context 131072
```

The helper verifies the loaded model/context before sending a probe and reports
strict JSON separately from exact-code presence. Prompt RNG seeds117/219 are
filler-generation seeds, not API sampling seeds. Fixed word counts and LF hashes
are supplied; do not recalibrate when reproducing the exact published fixtures.
Byte-identical prompt text does not imply identical adaptive-cache or OS state.

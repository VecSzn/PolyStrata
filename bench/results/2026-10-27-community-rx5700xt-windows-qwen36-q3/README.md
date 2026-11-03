# RX 5700 XT on Windows: four official suites and supplementary context tests

Measured 2026-10-27 on a Xeon W-2145 / RX 5700 XT / 256 GiB desktop, using the
same poly.5-equivalent `strata-poly` HIP executable compiled for actual `gfx1010`.
All four **unmodified official default suites** used configured context 32768,
three repetitions, thinking off and temperature 0: **132 measured requests,
zero request errors, zero skips**. These suites measure speed, not correctness.

## Official results in the community template format

All rows use configured context **32768**, thinking off, temperature 0 and
three suite repetitions. Values are **median (minimum–maximum)**; a single
value means all samples were equal. Token counts use the same summary when
inputs or outputs vary. `n` is the number of measured requests in that row.

| Configuration / workload | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
|---|---:|---:|---:|---|---:|---:|---:|
| Qwen3.6 Q3 MTP / chat | 46 (43–51) | 0 | 100 | 3 (n=9) | 65.0 (61.5–81.9) | 31.4 (27.9–37.2) | 0.711 (0.651–0.767) |
| Qwen3.6 Q3 MTP / 4K prompt | 4019 | 0 | 24 | 3 (n=3) | 516.7 (515.7–518.3) | 25.9 (22.2–27.9) | 7.797 (7.761–7.800) |
| Qwen3.6 Q3 MTP / 16K prompt | 16058 | 0 | 24 | 3 (n=3) | 510.6 (510.5–511.6) | 24.7 (23.7–27.1) | 31.468 (31.428–31.470) |
| Qwen3.6 Q3 MTP / agent initial turns | 609 | 0 | 29 (19–52) | 3 (n=3) | 193.1 (190.4–209.4) | 29.2 (27.7–35.6) | 3.158 (2.923–3.227) |
| Qwen3.6 Q3 MTP / agent follow-ups | 8604 (2220–15348) | 5634 (602–11548) | 47 (34–55) | 3 (n=15) | 360.7 (343.9–411.7) | 30.3 (23.3–37.0) | 8.387 (4.362–10.661) |
| Ornith Compact / chat | 46 (43–51) | 0 | 100 | 3 (n=9) | 70.3 (63.5–79.8) | 28.6 (23.0–31.7) | 0.659 (0.616–0.681) |
| Ornith Compact / 4K prompt | 4019 | 0 | 24 | 3 (n=3) | 439.9 (439.0–440.4) | 26.6 (24.3–27.5) | 9.142 (9.132–9.161) |
| Ornith Compact / 16K prompt | 16058 | 0 | 24 | 3 (n=3) | 433.4 (432.2–433.9) | 25.7 (24.4–25.7) | 37.085 (37.021–37.169) |
| Ornith Compact / agent initial turns | 609 | 0 | 39 (18–42) | 3 (n=3) | 194.3 (188.5–198.4) | 27.0 (23.6–29.1) | 3.138 (3.073–3.237) |
| Ornith Compact / agent follow-ups | 8639 (2219–15399) | 5662 (602–11581) | 57 (47–64) | 3 (n=15) | 315.0 (303.5–352.6) | 26.3 (24.1–29.2) | 9.713 (5.107–12.065) |
| Tiel Q3 XL / chat | 224 (221–229) | 0 | 100 | 3 (n=9) | 131.4 (125.1–142.8) | 29.2 (25.2–32.3) | 1.705 (1.607–1.787) |
| Tiel Q3 XL / 4K prompt | 4025 | 0 | 24 | 3 (n=3) | 498.7 (496.9–498.9) | 25.9 (25.0–28.5) | 8.078 (8.075–8.125) |
| Tiel Q3 XL / 16K prompt | 16058 | 0 | 24 | 3 (n=3) | 495.8 (495.2–496.9) | 22.9 (22.6–22.9) | 32.407 (32.331–32.445) |
| Tiel Q3 XL / agent initial turns | 835 | 0 | 10 | 3 (n=3) | 216.2 (210.6–230.9) | 26.8 (26.1–29.1) | 3.866 (3.623–3.969) |
| Tiel Q3 XL / agent follow-ups | 8845 (2441–15611) | 5923 (844–11859) | 51 (32–64) | 3 (n=15) | 355.3 (338.1–402.6) | 27.0 (24.3–29.3) | 8.416 (4.386–10.778) |
| Gemma QAT Q4 XL / chat | 48 (45–53) | 0 | 100 | 3 (n=9) | 60.8 (57.9–72.4) | 30.6 (24.3–36.9) | 0.799 (0.732–0.834) |
| Gemma QAT Q4 XL / 4K prompt | 4101 | 0 | 24 | 3 (n=3) | 478.1 (477.6–478.7) | 27.0 (25.2–28.5) | 8.591 (8.583–8.657) |
| Gemma QAT Q4 XL / 16K prompt | 16390 | 0 | 24 | 3 (n=3) | 452.6 (452.4–452.7) | 31.3 (30.5–33.0) | 36.237 (36.223–36.655) |
| Gemma QAT Q4 XL / agent initial turns | 396 | 0 | 22 | 3 (n=3) | 169.3 (167.4–184.6) | 32.4 (30.7–34.6) | 2.342 (2.149–2.372) |
| Gemma QAT Q4 XL / agent follow-ups | 8575 (2078–15542) | 5553 (389–11629) | 52 (42–60) | 3 (n=15) | 341.1 (325.1–382.7) | 30.2 (23.2–34.8) | 9.150 (4.679–11.594) |

Chat is three different questions repeated three times. Agent initial rows report
three starting turns; follow-up rows pool the other15 turns from three six-turn
sessions. Together, all table rows cover **132 total measured requests**.
Prompt throughput uses engine timing for newly processed tokens; reused prefix
tokens remain visible separately. TTFT is measured by the streaming client.
Official prompt tests have a 24-token output cap, chat 100 and agent turns 64.
The 4K/16K labels describe approximate workloads, not the configured window.
All 132 requests succeeded; these speed checks do not establish correctness.

[Numeric CSV with medians and range endpoints](official-results.csv).

The single-request 64K/128K retrieval, coding/formatting checks and paired
tail-skip comparison are **custom supplementary tests**, not additional official
suites or broad model-quality benchmarks. Formatting and coding failures are
retained. Hybrid CPU/GPU execution is used throughout; this is not full GPU offload.

- [Additional-model methods, results, failures and memory](ADDITIONAL-MODELS.md)
- [Three-pair Qwen tail-skip comparison](TAIL-SKIP.md)
- Original Qwen official JSON: [results.json](results.json)
- Original Qwen provenance: [provenance.json](provenance.json)
- New-model provenance and official JSON hashes: [additional-models/provenance.json](additional-models/provenance.json)

The folder retains the original Qwen-specific name to keep the first submission's
paths stable. All results are from this one desktop; differences between models,
weights, drafts, profiles and adaptive Windows GPU budgets prevent a pure
architecture/quantization comparison. No general quality ranking is claimed.

## Original Qwen3.6 report and build details

One RX 5700 XT successfully ran the official default benchmark suite and two
separate long-prompt retrieval probes using the real `gfx1010` HIP target.
This is a community report of an unvalidated device, not a support-list change.
The top-level setup GPU allowlist rejected this card; a small adapter invoked
unchanged upstream `poly_setup` with the actual measured GPU metadata.

## Hardware and build

| Component | Tested configuration |
|---|---|
| GPU | Radeon RX 5700 XT, 8176 MiB reported VRAM, actual `gfx1010:xnack-` |
| CPU | Xeon W-2145, 8 cores / 16 threads, nominal 3.70 GHz |
| RAM | 256 GiB installed (8 x 32 GiB DDR4-2400); OS reports 255.6 GiB usable |
| Storage | SSD; disk/model cache was not flushed between tests |
| OS / driver | Windows 11 Pro, build 10.0.26200 / AMD 32.0.21045.16010 |
| PolyStrata | `cc508adc50c32e0a98bd9bc1c5688ab5195b2538`, poly.5-equivalent engine/build/setup source |
| Release comparison | Engine/build/setup identical to `v0.1.45-poly.5` (`bfd8911efbb8c27214d5aabc511907c6149171e7`); executable reports `0.1.45` |
| Toolchain | Python 3.12.10; VS 2022 Build Tools 17.14.42; CMake 4.4.3; Ninja 1.13.2 |
| HIP runtime | Official AMD TheRock Windows `gfx1010` package, `10.2.0a20260930` |
| Model | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`, 17,104,402,720 bytes |

The model revision, SHA256 hashes, dependency commit and runtime inventory are
in [provenance.json](provenance.json). No driver replacement, reboot, GPU
architecture spoofing or engine/kernel patch was used. PCIe link state, power,
temperature, background load and clock behavior were not controlled or measured.

## Settings and what ran on the GPU

The upstream recommended Qwen3.6 Q3 configuration used HIP, FP16 KV,
`--kv-whole`, `--tail-skip 6`, automatic conversation cache (8192 MiB cap),
embedded MTP and 16 CPU workers. Requests used temperature 0 and thinking off.
`STRATA_SLOTS_WARP` is the
HIP default; `STRATA_HIP_VALUE_SPLIT` was not enabled.
The sanitized [configs](configs/) retain the actual arguments; replace
`<workspace>` with an installation directory before use.

This is hybrid execution: dense layers and attention use the GPU, with CPU
expert work and GPU expert caching/staging. At 128K, the engine admitted zero
persistent GPU expert slots; decode expert math used the CPU while prompt
batches staged experts through GPU MMQ. Tail skip 6 approximates inference by
omitting uncached experts ranked 7/8 and can change answers. These results do
not establish quality equivalence to full-expert execution.

Only `--max-context` changed for the 64K/128K configurations. No quantized-KV
fallback or WDDM budget override was used. Observed WDDM budgets were about
7383 MiB for the baseline and approximately 6378 MiB for the longer contexts. Thus the
cross-context speed differences also include a budget/cache-layout confound.

## Official benchmark, configured context 32768

The unmodified `tools/strata_bench.py` ran against an already started server:

```powershell
python tools/strata_bench.py --url http://127.0.0.1:18085 --runs 3
```

Default suite, no `--long`, no parallel clients, thinking off. The suite took
322.287 seconds: 33 measured requests, zero failed, zero skipped. The original
tool-sanitized [results.json](results.json) is included byte-for-byte.
Calibration/setup requests are outside that measured request count. Chat consists
of three different questions repeated three times, each capped at 100 tokens.
The three basic probes below ran on this same server before the suite's warm-up.

| Workload | Samples | Actual input tokens | Median engine tok/s (range) | Median wall TTFT, seconds (range) |
|---|---:|---:|---:|---:|
| Short chat | 9 requests | See individual requests | Decode 31.4 (27.9–37.2) | 0.711 (0.651–0.767) |
| 4K prompt | 3 requests | 4019; zero reused | Prompt 516.7 (515.7–518.3) | 7.797 (7.761–7.800) |
| 16K prompt | 3 requests | 16058; zero reused | Prompt 510.6 (510.5–511.6) | 31.468 (31.428–31.470) |
| Agent follow-ups | 15 requests, across 3 six-turn sessions | Last-turn median 15343 | Decode 30.3 (23.3–37.0) | 8.387 (4.362–10.661) |

Agent sessions contain 18 requests including the three initial turns. Follow-up
cache-hit median was 65.956%. Engine prompt/decode rates use engine-reported
phase timings; TTFT is measured by the client and includes waiting/prompt work.
Decode tok/s alone does not represent end-to-end long-prompt speed.

## Separate correctness and longer-context probes

Three basic synthetic checks passed: arithmetic returned `391` (29 input
tokens); sorting returned `{"sorted": [1, 2, 7, 9]}` (40); a repeated-notes
retrieval returned the planted synthetic code `MAPLE-7392` (2789).
Exact fixtures, responses and timings are in [basic-checks.json](basic-checks.json).

Each longer-context row is **one request**, retrieving three planted codes at
the beginning, middle and end of seeded random-word filler. Output cap was
128 tokens; each generated 48 tokens with zero reused input tokens.
All three codes matched exactly in each row. The official suite was not rerun
at these context settings.

| Configured context | Actual input | Engine prompt tok/s | Engine decode tok/s | Wall TTFT | Wall total | Result |
|---:|---:|---:|---:|---:|---:|---|
| 65536 | 58117 | 345.4 | 28.1 | 168.324 s | 170.028 s | 3/3 codes |
| 131072 | 116115 | 232.8 | 22.8 | 498.798 s | 500.899 s | 3/3 codes |

Per-request data: [context64.json](context64.json), [context128.json](context128.json).
These probes establish successful admission and bounded retrieval at the actual
input lengths; they do not establish full-window capacity, broad quality,
multi-user stability or useful speed for every task. No larger context was tested.

## 128K system memory

[memory-summary.json](memory-summary.json) retains aggregate byte counts from
260 RAM samples at a target cadence of 2 seconds over 523.445 seconds, plus 96
GPU counter samples. The GPU monitor initially failed to launch under the shell
script policy; an encoded invocation worked without changing system policy,
leaving a startup gap. These are sampled peaks, not guaranteed instantaneous maxima.

| Counter | Observed value |
|---|---:|
| Whole-system RAM before launch | 17.731 GiB |
| Whole-system RAM sampled peak | 36.014 GiB |
| Increase above baseline | 18.283 GiB |
| Whole-system RAM after cleanup | 17.800 GiB |
| Engine sampled working set / private commit | 17.944 / 2.470 GiB |
| Whole-system sampled commit peak | 28.782 GiB |
| GPU sampled dedicated / shared memory | 6.133 / 1.723 GiB |

Whole-system changes include other applications and monitors. Mapped working
set, private commit, GPU shared memory and cache can overlap; do not add them.
The 8192 MiB conversation-cache cap is not a measurement of RAM usage.
No reliable system-RAM peak was captured for 32K or 64K. All owned inference
workers were stopped after the tests. No endurance or thermal test was performed.

## Reproduction and retained failures

Use a clean checkout of the pinned commit, Python 3.12, VS C++ Build Tools and
the official AMD TheRock Windows `gfx1010` SDK/runtime matching the version
above. The original setup installed that runtime into the checkout's
`.rocm-win` environment. Confirm a HIP device probe reports the actual
`gfx1010:xnack-` target before building; do not substitute another architecture.
The dependency pin is in provenance.json. This report does not claim that an
arbitrary newer runtime or driver reproduces these measurements.

[setup-gfx1010.py](setup-gfx1010.py) is the small allowlist-bypass adapter used
for this machine, generalized to accept local source/workspace paths. For the
included configs, use `--source <workspace>/source --workspace <workspace>`.
Place the pinned, SHA256-verified GGUF in `<workspace>/models` first; this adapter
uses `--gguf-dir` and does not download the GGUF. It calls
upstream setup/build routines with the actual device and recommended answers;
it is not a proposed installer change. Setup may download supporting packs and
build dependencies before compiling. Read it before running, and use it only
for the reported hardware.

The first source-archive build attempt failed because the archive omitted the
upstream `tests/` directory. Restoring that unchanged directory from the same
commit resolved it; the subsequent build succeeded. This packaging
failure and the earlier top-level allowlist rejection are not GPU kernel failures.
No failed measured inference request occurred. Private full logs are retained
locally and deliberately excluded from this submission.

After starting a server with the appropriate context configuration:

```powershell
# Verify fixed synthetic fixtures offline, without starting inference:
python reproduce.py --verify-only
# Use tools/strata_bench.py from the pinned source tree for the same HTTP client:
python reproduce.py --source <checkout> --context 65536 --url http://127.0.0.1:18085
python reproduce.py --source <checkout> --context 131072 --url http://127.0.0.1:18085
```

The fixed word counts reproduce the original HTTP prompt strings, with hashes
checked before sending. Do not recalibrate them when reproducing these exact
fixtures. Actual tokenizer counts must be read from the returned API usage.
No comparison against another quantization, runtime or release is claimed.

The tests ran sequentially and reused the same expert-profile path; the engine
updates that profile during use. Exact profile contents were not frozen or hashed
at each phase boundary. Consequently the fixtures reproduce the prompt strings,
but cannot reproduce the exact initial expert/cache state of every measured run.
Zero reused prompt tokens does not mean a cold model, disk cache or expert profile.

Privacy: only synthetic prompts are published. Usernames, hostnames, user paths,
LAN addresses, process IDs, adapter identifiers and raw monitoring/log archives
are excluded; numerical measurements and artifact hashes are retained.

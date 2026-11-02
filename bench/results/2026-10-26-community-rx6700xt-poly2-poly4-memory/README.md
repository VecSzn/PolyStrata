# RX 6700 XT 12 GB: poly.2 / poly.4 and memory availability

Measured on **2026-10-25 and 2026-10-26**. Twenty-eight official `tools/strata_bench.py` suites compare **v0.1.45-poly.2** with **v0.1.45-poly.4**, at 28 and 52 GiB inference-container limits on the same upgraded 64 GB machine. All use a 65,536-token configured window. This is a dated follow-up to [the October 19 report](../2026-10-19-community-rx6700xt-polystrata/README.md), merged in [PR #52](https://github.com/VecSzn/PolyStrata/pull/52); the earlier measurements are preserved.

## Exact tested versions

Both local HIP binaries report numeric engine version `0.1.45`; the release tag, full commit and engine hash distinguish them. The same source checkouts provide the corresponding server bundles. These measurements apply to these builds, not future releases.

| Release | Source commit | Native gfx1031 engine SHA-256 |
| --- | --- | --- |
| [v0.1.45-poly.2](https://github.com/VecSzn/PolyStrata/releases/tag/v0.1.45-poly.2) | `974552f562aecb3e86cdd2679e76e4515b52b9d2` | `5cecdf2d8354ae704e456914c8d453d3497dfa8200b52e6c0f61802d52912aca` |
| [v0.1.45-poly.4](https://github.com/VecSzn/PolyStrata/releases/tag/v0.1.45-poly.4) | `145493e66580c80bba09b0130f667a2b1789da69` | `372c6ed707d302c470b864a8fc24ff2a3e462f94cb33c0956c20205a95ef9785` |

ROCm userspace is **7.13.0a20260515**; llama.cpp/ggml dependency is pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Both builds reuse the same compiler/libraries. See [software.json](software.json) and [BUILD.txt](BUILD.txt). No ready-made CUDA binary or architecture override was used.

## Hardware and limits

- AMD Radeon RX 6700 XT, 12 GB dedicated VRAM, `gfx1031`; one GPU. Ryzen 5 3600, 6 physical cores / 12 logical CPUs; eight engine workers.
- 64 GB nominal DDR4, four 16 GB DIMMs configured at 3200 MT/s (two rank-1 and two rank-2). Host-reported usable RAM is about 62.7 GiB. These physical DIMMs stay unchanged throughout the new comparison.
- Debian 13 in an unprivileged LXC limited to 56 GiB, no guest swap; host kernel `7.0.14-24-pve` and amdgpu. Python 3.13.5.
- Inference Docker limits are 28 or 52 GiB, a nine-CPU quota, nonroot and offline. Docker MemorySwap equals Memory, allowing no additional container swap. Client limit is one CPU / 1.5 GiB. Actual Docker limits and non-OOM completion were checked per suite.
- Existing Intel SSDPEKNW010T8 NVMe storage. A post-run metadata check exposed a PCIe link of 16.0 GT/s x16 and a 211 W GPU power cap; these are snapshots, not continuous clock/power measurements. No power-limit change was made for this matrix.
- The upstream AMD detector leaves GPU fields empty and generates `cpu` labels. Its RAM/CPU fields can reflect host/container visibility rather than effective quotas. Those emitted values remain unchanged; the verified device and explicit limits above identify the actual test environment.

**The 28/52 GiB factor measures available memory capacity on the upgraded machine. It does not recreate the old 32 GB physical DIMMs, their ranks or any memory-bandwidth changes.** The October 19 report used a 26 GiB inference container inside a 28 GiB LXC, plus different source/configuration/workloads. Cross-date changes are observational and are not isolated RAM-upgrade gains.

## Workload and reproducibility

The unmodified official client and its document corpus were frozen from poly.2 before the matrix. Its SHA-256 is `bd65126bed1f8e5ff3648f5d65c230ea7566219a443272115c1d622735d02be5`, also matching the poly.4 client. [official-client-freeze.json](official-client-freeze.json) records all frozen files; use the poly.2 checkout for the corpus even when running the poly.4 server.

Each configuration starts a fresh server and short readiness requests, then runs three repetitions of the upstream short-chat, fresh-prompt (~4K/~16K/~32K) and six-turn agent workloads. The suite discards its warm-up chat and 1K prompt. Model loading is outside throughput timings. Fresh prompts carry unique tags; the agent phase intentionally reuses prefixes. No simultaneous-client phase was requested.

The same initial expert-profile seed is cloned into all four cells of each model configuration; the seven exact seeds and hashes are included. Adaptive placement within each running suite is retained. Order is profile, then 28 GiB poly.2/poly.4, then 52 GiB poly.4/poly.2. This is a sequential experiment without randomized repeats or significance estimates.

All configurations use `--kv-whole`, eight workers, 1024 MiB VRAM reserve and zero parked conversation cache. LFM adds `--prompt-batch 2048`. Qwen Q3 retains approximate `--tail-skip 6`; Qwen Q4 control uses `--tail-skip 0`. Family draft behavior remains enabled; Gemma supplies its Q8_0 draft. Thinking is disabled by the official client; Gemma also records server reasoning mode off. `benchmark_mtp=0` is an adapter sentinel, not a switch disabling engine speculation. Vision is disabled.

Each run includes the exact sanitized server `config.json`, official report, all request timings and a supplementary provenance file with release, binary, model and initial-profile hashes. [models.json](models.json) records pinned installer repositories/revisions and previously verified full-file weight hashes; the same weights were reused through read-only mounts. No large weights or generated packs are included.

With the selected server already running and paths in its config replaced for your installation, invoke the frozen client:

```sh
python tools/strata_bench.py --url http://127.0.0.1:8080 --config CONFIG.json --runs 3 --long --out OUTPUT
```

Throughput is from engine timings; TTFT is measured by the streaming client. Agent speed summarizes follow-up turns, not the initial turn. Every result JSON retains medians, ranges, per-request token counts, prefix reuse, draft acceptance and failures. Failed requests are outside successful throughput summaries.

## Measurements

All throughput values below are median tokens/s; TTFT is seconds. The full [CSV](summary.csv) also contains dates, actual input maxima, agent TTFT, skips and source/published hashes. Allocated 65,536 context does not mean 65,536 input tokens were exercised.

| Configuration | Release | RAM limit GiB | Chat | Chat TTFT | Prefill ~4K | Prefill ~16K | Prefill ~32K | Agent | Max input | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Gemma 4 UD-Q4_K_M](runs/gemma4-ud-q4_k_m-reserve1024-thinkingoff-poly2-ram28-ctx65536/official-report.md) | v0.1.45-poly.2 | 28 | 70.8 | 0.566 | 921.2 | 983.0 | 819.8 | 70.3 | 32,774 | 0 |
| [Gemma 4 UD-Q4_K_M](runs/gemma4-ud-q4_k_m-reserve1024-thinkingoff-poly2-ram52-ctx65536/official-report.md) | v0.1.45-poly.2 | 52 | 69.0 | 0.581 | 891.7 | 974.7 | 815.4 | 65.8 | 32,774 | 0 |
| [Gemma 4 UD-Q4_K_M](runs/gemma4-ud-q4_k_m-reserve1024-thinkingoff-poly4-ram28-ctx65536/official-report.md) | v0.1.45-poly.4 | 28 | 75.2 | 0.585 | 891.9 | 970.2 | 812.5 | 66.5 | 32,774 | 0 |
| [Gemma 4 UD-Q4_K_M](runs/gemma4-ud-q4_k_m-reserve1024-thinkingoff-poly4-ram52-ctx65536/official-report.md) | v0.1.45-poly.4 | 52 | 70.7 | 0.590 | 895.5 | 969.5 | 812.8 | 69.8 | 32,774 | 0 |
| [LFM2.5 UD-Q4_K_XL](runs/lfm25-ud-q4_k_xl-reserve1024-poly2-ram28-ctx65536/official-report.md) | v0.1.45-poly.2 | 28 | 78.0 | 0.215 | 1523.0 | 1476.7 | 1118.0 | 77.0 | 32,875 | 0 |
| [LFM2.5 UD-Q4_K_XL](runs/lfm25-ud-q4_k_xl-reserve1024-poly2-ram52-ctx65536/official-report.md) | v0.1.45-poly.2 | 52 | 79.9 | 0.222 | 1545.3 | 1481.1 | 1119.5 | 72.1 | 32,875 | 0 |
| [LFM2.5 UD-Q4_K_XL](runs/lfm25-ud-q4_k_xl-reserve1024-poly4-ram28-ctx65536/official-report.md) | v0.1.45-poly.4 | 28 | 78.4 | 0.221 | 1546.0 | 1480.7 | 1120.8 | 70.3 | 32,875 | 2 |
| [LFM2.5 UD-Q4_K_XL](runs/lfm25-ud-q4_k_xl-reserve1024-poly4-ram52-ctx65536/official-report.md) | v0.1.45-poly.4 | 52 | 78.1 | 0.217 | 1546.8 | 1477.8 | 1120.9 | 68.2 | 32,875 | 2 |
| [MiniCPM-V mix_q3gud_im](runs/minicpmv-mix_q3gud_im-reserve1024-poly2-ram28-ctx65536/official-report.md) | v0.1.45-poly.2 | 28 | 59.2 | 0.344 | 1094.9 | 1001.5 | 789.0 | 54.5 | 32,049 | 0 |
| [MiniCPM-V mix_q3gud_im](runs/minicpmv-mix_q3gud_im-reserve1024-poly2-ram52-ctx65536/official-report.md) | v0.1.45-poly.2 | 52 | 59.3 | 0.353 | 1101.9 | 1005.7 | 788.1 | 54.6 | 32,049 | 0 |
| [MiniCPM-V mix_q3gud_im](runs/minicpmv-mix_q3gud_im-reserve1024-poly4-ram28-ctx65536/official-report.md) | v0.1.45-poly.4 | 28 | 59.0 | 0.368 | 1102.3 | 1011.1 | 791.5 | 53.9 | 32,049 | 0 |
| [MiniCPM-V mix_q3gud_im](runs/minicpmv-mix_q3gud_im-reserve1024-poly4-ram52-ctx65536/official-report.md) | v0.1.45-poly.4 | 52 | 59.1 | 0.380 | 1104.4 | 1011.0 | 796.0 | 54.9 | 32,049 | 0 |
| [Ornith APEX-I-Compact](runs/ornith-apex-i-compact-reserve1024-poly2-ram28-ctx65536/official-report.md) | v0.1.45-poly.2 | 28 | 58.6 | 0.362 | 1019.5 | 979.3 | 779.1 | 54.3 | 32,049 | 0 |
| [Ornith APEX-I-Compact](runs/ornith-apex-i-compact-reserve1024-poly2-ram52-ctx65536/official-report.md) | v0.1.45-poly.2 | 52 | 64.4 | 0.361 | 1026.6 | 982.4 | 777.5 | 57.8 | 32,049 | 0 |
| [Ornith APEX-I-Compact](runs/ornith-apex-i-compact-reserve1024-poly4-ram28-ctx65536/official-report.md) | v0.1.45-poly.4 | 28 | 65.2 | 0.389 | 1018.8 | 980.8 | 781.9 | 57.5 | 32,049 | 0 |
| [Ornith APEX-I-Compact](runs/ornith-apex-i-compact-reserve1024-poly4-ram52-ctx65536/official-report.md) | v0.1.45-poly.4 | 52 | 63.8 | 0.376 | 1022.1 | 984.1 | 784.5 | 58.2 | 32,049 | 0 |
| [Qwen3.6 UD-Q3_K_M](runs/qwen36-ud-q3_k_m-reserve1024-poly2-ram28-ctx65536/official-report.md) | v0.1.45-poly.2 | 28 | 77.2 | 0.397 | 1074.4 | 1076.5 | 846.1 | 70.1 | 32,049 | 0 |
| [Qwen3.6 UD-Q3_K_M](runs/qwen36-ud-q3_k_m-reserve1024-poly2-ram52-ctx65536/official-report.md) | v0.1.45-poly.2 | 52 | 77.1 | 0.408 | 1073.9 | 1082.9 | 841.3 | 69.1 | 32,049 | 0 |
| [Qwen3.6 UD-Q3_K_M](runs/qwen36-ud-q3_k_m-reserve1024-poly4-ram28-ctx65536/official-report.md) | v0.1.45-poly.4 | 28 | 76.6 | 0.423 | 1070.2 | 1086.2 | 851.7 | 68.1 | 32,049 | 0 |
| [Qwen3.6 UD-Q3_K_M](runs/qwen36-ud-q3_k_m-reserve1024-poly4-ram52-ctx65536/official-report.md) | v0.1.45-poly.4 | 52 | 76.0 | 0.430 | 1073.8 | 1080.2 | 850.5 | 69.6 | 32,049 | 0 |
| [Qwen3.6 UD-Q4_K_M control](runs/qwen36-ud-q4_k_m-control-reserve1024-poly2-ram28-ctx65536/official-report.md) | v0.1.45-poly.2 | 28 | 77.2 | 0.454 | 906.4 | 1014.8 | 821.9 | 76.1 | 32,049 | 0 |
| [Qwen3.6 UD-Q4_K_M control](runs/qwen36-ud-q4_k_m-control-reserve1024-poly2-ram52-ctx65536/official-report.md) | v0.1.45-poly.2 | 52 | 77.5 | 0.477 | 907.0 | 1019.0 | 816.8 | 75.6 | 32,049 | 0 |
| [Qwen3.6 UD-Q4_K_M control](runs/qwen36-ud-q4_k_m-control-reserve1024-poly4-ram28-ctx65536/official-report.md) | v0.1.45-poly.4 | 28 | 77.7 | 0.469 | 905.2 | 1020.6 | 823.0 | 76.5 | 32,049 | 0 |
| [Qwen3.6 UD-Q4_K_M control](runs/qwen36-ud-q4_k_m-control-reserve1024-poly4-ram52-ctx65536/official-report.md) | v0.1.45-poly.4 | 52 | 78.7 | 0.470 | 908.5 | 1025.7 | 821.9 | 76.4 | 32,049 | 0 |
| [Tiel Coder UD-Q3_K_XL](runs/tiel-ud-q3_k_xl-reserve1024-poly2-ram28-ctx65536/official-report.md) | v0.1.45-poly.2 | 28 | 66.2 | 0.737 | 1090.2 | 1063.3 | 841.3 | 60.2 | 32,049 | 0 |
| [Tiel Coder UD-Q3_K_XL](runs/tiel-ud-q3_k_xl-reserve1024-poly2-ram52-ctx65536/official-report.md) | v0.1.45-poly.2 | 52 | 67.0 | 0.729 | 1105.5 | 1074.6 | 840.1 | 61.3 | 32,049 | 0 |
| [Tiel Coder UD-Q3_K_XL](runs/tiel-ud-q3_k_xl-reserve1024-poly4-ram28-ctx65536/official-report.md) | v0.1.45-poly.4 | 28 | 68.6 | 0.734 | 1098.9 | 1077.7 | 842.3 | 60.9 | 32,049 | 0 |
| [Tiel Coder UD-Q3_K_XL](runs/tiel-ud-q3_k_xl-reserve1024-poly4-ram52-ctx65536/official-report.md) | v0.1.45-poly.4 | 52 | 67.0 | 0.750 | 1105.6 | 1077.4 | 846.9 | 61.0 | 32,049 | 0 |

The Qwen3.6 Q3 short-chat medians are 77.2 / 76.6 tok/s at 28 GiB and 77.1 / 76.0 at 52 GiB (poly.2 / poly.4). Gemma is 70.8 / 75.2 at 28 GiB and 69.0 / 70.7 at 52 GiB. Ornith varies more: 58.6 / 65.2 at 28 GiB and 64.4 / 63.8 at 52 GiB. These mixed results do not support a general speedup from either factor across the tested models. Larger models that need the extra RAM are outside this report.

## Failures and scope

The LFM poly.4 suite records **two agent requests with no token returned at each RAM limit**, four total. They remain in the reports and JSON; these are not successful zero-throughput observations. The other 26 suites record no request failures. All 28 suites have zero skipped tests. All suites completed without Docker reporting an OOM kill.

A poly.4 compatibility check on this card preceded measurement and exercised repeated API requests, streaming and a structured tool call/result. This is a compatibility gate, not a general tool-use accuracy score.

This submission contains only this completed official speed matrix. Separate V3.1 quality/business diagnostics, new-capacity models and the scheduled Qwen 128K appendix remain outside its scope. No full-64K comprehension, general model-accuracy, production-reliability, llama.cpp or power-efficiency claim is made. Family quantization/draft differences and approximate Qwen Q3 settings prevent equal-quality cross-model comparisons.

## Data integrity and privacy

Installation-path strings were replaced by neutral placeholders in the official outputs. JSON keys, all non-string values and the entire inference `results` subtree were checked against the frozen originals and are unchanged. [data-integrity.json](data-integrity.json) records original and published result hashes. Missing AMD detector fields and missing original commit fields were not silently filled; supplementary provenance supplies the externally verified identity.

[MANIFEST.json](MANIFEST.json) lists every submitted artifact and its SHA-256. Private machine names, account names, LAN addresses, credentials, orchestration logs and custom task outputs are excluded. Loopback API settings are retained.

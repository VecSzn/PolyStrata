# RTX 3060 12 GB: PolyStrata v0.1.45-poly.2 and v0.1.45-poly.3

Measured on **2026-10-22 and 2026-10-24** on the RTX 3060 desktop. This is a historical report of the two releases named below. **v0.1.45-poly.4 was already available when this report was submitted; it was not tested here. These are not latest-release results.**

## Tested software

Both engines report the numeric version `0.1.45`. The release suffix, full source commit and binary hash distinguish the tested builds.

| Release | Source commit | Windows engine |
| --- | --- | --- |
| [v0.1.45-poly.2](https://github.com/VecSzn/PolyStrata/releases/tag/v0.1.45-poly.2) | [`974552f562aecb3e86cdd2679e76e4515b52b9d2`](https://github.com/VecSzn/PolyStrata/tree/974552f562aecb3e86cdd2679e76e4515b52b9d2) | Official CUDA 13.0 release binary |
| [v0.1.45-poly.3](https://github.com/VecSzn/PolyStrata/releases/tag/v0.1.45-poly.3) | [`f096adb0c2739b6348a5c69f94dfb09737c4b044`](https://github.com/VecSzn/PolyStrata/tree/f096adb0c2739b6348a5c69f94dfb09737c4b044) | Official CUDA 13.0 release binary |

Engine SHA-256 values and client identity are in [software.json](software.json). Every run has a supplementary `provenance.json` with its release, commit and binary hash. Original missing `strata_commit` fields remain missing in the tool output.

Official client: [`tools/strata_bench.py` at poly.3](https://github.com/VecSzn/PolyStrata/blob/f096adb0c2739b6348a5c69f94dfb09737c4b044/tools/strata_bench.py), SHA-256 `bd65126bed1f8e5ff3648f5d65c230ea7566219a443272115c1d622735d02be5`. The archived poly.2 client has the same hash. No benchmark-code changes.

## Hardware and method

- NVIDIA GeForce RTX 3060, 12,288 MiB VRAM; Intel Xeon W-2145, 8 cores / 16 threads; 32 GB installed RAM (31.6 GiB reported), DDR4-2666.
- Windows 11 Pro for Workstations, build 26200; Python 3.12.10. NVIDIA driver **591.59 on October 24**, **610.88 on October 22**. Engine CUDA build version is 13.0; the October 24 driver reports CUDA support 13.1, which is a different field.
- Thirteen official suites, three repetitions each, thinking off, no concurrent-client phase. Models ran serially. Downloads and model loading are outside reported inference timings.
- The nine main October 24 suites use a 32,768-token allocation and an 8,192 MiB parked conversation cache. Family defaults, native drafts and adaptive expert placement remain enabled. Qwen uses approximate `--tail-skip 6`.
- The two context follow-ups use 65,536 / 131,072 allocations, zero parked conversation cache, fp16 KV and `--vram-reserve-mib 1024`. They are different complete configurations from the 32K runs.
- Engine timings supply prefill/decode throughput; client streaming timestamps supply time to first token (TTFT). Tables show medians. Each original `results.json` retains repetitions, ranges, token counts, failures, settings, launch command, engine information and cache data.
- Fresh-prompt tests use the official suite's unique tags. Agent follow-ups intentionally reuse conversation state. Adaptive expert-cache starting state can vary; this was a sequential desktop experiment, not a randomized controlled study.
- PCIe link state, storage throughput, power limits, clocks and background desktop load were not controlled or recorded in this submission. No hardware-upgrade or power-efficiency claim is made.

Run from the corresponding release checkout. Save `machine.config` from the selected result as `run-config.json`, replacing neutral file paths with your installed paths:

```powershell
python tools/strata_bench.py --config run-config.json --runs 3
# Context follow-ups only:
python tools/strata_bench.py --config run-config.json --runs 3 --long
```

The per-run `machine.config` and `machine.launch` give the exact engine flags. Model/draft filenames, pinned repository revisions, byte sizes and full SHA-256 values are in [models.json](models.json). Expert-profile paths are neutral placeholders; adaptive profile contents were not included, so the initial expert placement is not exactly reproducible.

For the separately measured MMA pair, `STRATA_GDN_MMA` was unset in the off run and set to `1` in the on run; all other runs left it unset. Environment overrides are recorded per run.

## October 24: 32K allocation, driver 591.59

All throughput columns are tokens/s. TTFT is seconds. Qwen = Qwen3.6-35B-A3B UD-Q3_K_M; Gemma = Gemma 4 26B-A4B UD-Q4_K_M; Tiel = Tiel Coder 35B-A3B UD-Q3_K_XL; LFM = LFM2.5-24B-A2B UD-Q4_K_XL; Flint = Flint 40B-A2.5B UD-Q3_K_XL. See the model manifest for full names.

| Model | Tested release | Variant | Chat | Chat TTFT | Prefill ~4K | Prefill ~16K | Agent | Failed requests |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiel ([raw](runs/tiel-ud-q3_k_xl__v0.1.45-poly.3__default/results.json)) | v0.1.45-poly.3 | default | 78.8 | 0.961 | 1243.0 | 1425.9 | 73.3 | 0 |
| LFM ([raw](runs/lfm25-ud-q4_k_xl__v0.1.45-poly.3__default/results.json)) | v0.1.45-poly.3 | default | 97.8 | 0.219 | 2173.3 | 3140.8 | 83.5 | 2 |
| Flint ([raw](runs/flint-ud-q3_k_xl__v0.1.45-poly.3__default/results.json)) | v0.1.45-poly.3 | default | 65.5 | 0.292 | 1515.2 | 1935.9 | 61.1 | 0 |
| Qwen ([raw](runs/qwen36-ud-q3_k_m__v0.1.45-poly.3__default/results.json)) | v0.1.45-poly.3 | default | 91.8 | 0.440 | 1258.7 | 1417.9 | 79.1 | 0 |
| Gemma ([raw](runs/gemma4-ud-q4_k_m__v0.1.45-poly.3__default/results.json)) | v0.1.45-poly.3 | default | 78.0 | 0.475 | 1269.2 | 1654.2 | 65.4 | 0 |
| Qwen ([raw](runs/qwen36-ud-q3_k_m__v0.1.45-poly.2__default/results.json)) | v0.1.45-poly.2 | default | 85.8 | 0.418 | 1201.6 | 1321.3 | 75.6 | 0 |
| Gemma ([raw](runs/gemma4-ud-q4_k_m__v0.1.45-poly.2__default/results.json)) | v0.1.45-poly.2 | default | 75.0 | 0.449 | 1213.4 | 1596.6 | 65.2 | 0 |
| Qwen ([raw](runs/qwen36-ud-q3_k_m__v0.1.45-poly.3__mma_on/results.json)) | v0.1.45-poly.3 | mma_on | 89.7 | 0.403 | 1446.9 | 1675.4 | 79.5 | 0 |
| Qwen ([raw](runs/qwen36-ud-q3_k_m__v0.1.45-poly.3__mma_off/results.json)) | v0.1.45-poly.3 | mma_off | 88.8 | 0.413 | 1286.5 | 1442.7 | 79.5 | 0 |

The same-day default Qwen pair measured 85.8 → 91.8 chat tokens/s (+7.0%) from poly.2 to poly.3; Gemma measured 75.0 → 78.0 (+4.0%). These are observed medians without significance estimates. The separate Qwen MMA pair measured +12.5% / +16.1% prefill at ~4K / ~16K; chat changed +1.0%, agent median was unchanged. Quality parity was not tested in this report.

**LFM had two agent requests return no tokens.** They remain in the raw failures and are not counted as successful zero-throughput samples. Every other suite recorded zero failed requests. All thirteen suites recorded zero skips. Speed alone does not establish correct output or a model ranking; quantization, draft behavior and approximation settings differ.

## Context follow-ups: allocated window versus actual input

These October 24 runs use **v0.1.45-poly.3**, driver 591.59, Qwen UD-Q3_K_M and the official `--long` option. That option adds a roughly 32K prompt; it does not fill the configured window.

| Allocated context | Largest actual input | Chat tok/s | Prefill ~32K tok/s | Agent tok/s | Raw |
| ---: | ---: | ---: | ---: | ---: | --- |
| 65,536 | 32,049 | 90.9 | 1305.9 | 78.4 | [results](runs/qwen36-official-ctx65536/results.json) |
| 131,072 | 32,049 | 89.8 | 1305.9 | 74.2 | [results](runs/qwen36-official-ctx131072/results.json) |

**Neither row is evidence of processing a full 64K or 128K input.** Separate custom quality/retrieval tests are outside this official-speed submission. Context allocation plus output headroom differs from actual prompt length.

## October 22 historical appendix

Both rows use **v0.1.45-poly.2**, context 32,768, driver **610.88**. These are earlier measurements, not the October 24 same-day baselines. Date, driver and cache state confound cross-date comparisons; the reason for the driver change was not investigated.

| Model | Chat tok/s | Chat TTFT s | Prefill ~4K tok/s | Prefill ~16K tok/s | Agent tok/s | Raw |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen | 84.9 | 0.470 | 1222.8 | 1357.9 | 73.4 | [results](runs/qwen36-ud-q3_k_m__20261022__historical/results.json) |
| Gemma | 69.8 | 0.509 | 1238.5 | 1633.3 | 60.9 | [results](runs/gemma4-ud-q4_k_m__20261022__historical/results.json) |

## Files and verification

[summary.csv](summary.csv) contains all thirteen suites with release, date, driver, context, failures and original result hashes. [MANIFEST.json](MANIFEST.json) lists file sizes and SHA-256 values. The official `results.json` files are byte-for-byte copies of the tool's already-sanitized outputs, independently privacy-scanned before submission. No private machine names, user paths, network addresses, credentials or custom task outputs are included. Loopback listener settings are retained.

Only official speed measurements are included. No latest-release, llama.cpp, equal-quality, full-window comprehension or hardware-upgrade conclusion follows from this report.

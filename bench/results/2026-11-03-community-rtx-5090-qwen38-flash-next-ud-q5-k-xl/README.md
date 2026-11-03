# Community benchmark on GeForce RTX 5090 (Linux, 184 GiB)

Measured on 2026-10-31 and 2026-11-03 by @cpacsm75. Qwen3.8-Flash-Next UD-Q5_K_XL on a single
RTX 5090 in a headless Linux host, comparing `v0.1.45-poly.6` (20d839d) with `v0.1.45-poly.7`
(fabae0d) at 256K, 512K and 1M. One run per point except 512K (two runs, both listed); no
correctness evaluation, so these are speed measurements only.

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090, 32 GiB (sm_120a); PCIe 5.0 x16
- CPU: AMD Ryzen 7 9800X3D; RAM: 183 GiB DDR5-6000 (4×48 GB, full bank); storage: ext4 on NVMe (1.8 TB)
- OS: Ubuntu 26.04, headless (`multi-user.target`, no desktop session); driver 595.99.02
- CUDA 13.1 toolkit (unpacked to a prefix, not distro packages)
- Strata: v0.1.45-poly.6 (`20d839d`) and v0.1.45-poly.7 (`fabae0d`), both built from source
  with `CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=120`, `STRATA_ENABLE_CUDA=ON`
- Background workloads: none — each engine is the only GPU process during a measurement
- **GPU power limit: 450 W** (default 575 W; held at 450 W for cable temperature safety)

## Model and configuration

- `unsloth/Qwen3.8-Flash-Next-GGUF` — UD-Q5_K_XL, 158.3 GB (six shards), with the run's own pack
- Tokenizer from the pack; MTP enabled (`--mtp`, `--spec 4 --spec-min-p 0.5`)
- Context 1,048,576; KV `int8`; `--kv-resident 32768`; `--expert-cache auto`; `--prefill auto`
- `--vram-reserve-mib 700`, `--conversation-cache-mib 8192`, `--conversation-cache-slots 4`
- Environment: `STRATA_MMQ_KQ=1`, `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

```text
engine/strata --pack /data/PolyStrata-data/packs/unsloth-ud-q5_k_xl \
  --native /data/PolyStrata-data/models/unsloth-UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf \
  --expert-profile /data/PolyStrata/data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.5 --mtp /data/PolyStrata-data/mtp/rt \
  --max-context 1048576 --kv int8 --kv-resident 32768 --vision \
  --vram-reserve-mib 700 --conversation-cache-mib 8192 --conversation-cache-slots 4
```

## Method

`.venv/bin/python polystrata.py benchmark --model qwen3.8-flash-next-unsloth-ud-q5_k_xl --port 20080 --prompt-tokens <N>`
with `N` = 200,000 / 400,000 / 750,000. `--prompt-tokens` is a 4-characters-per-token estimate, so the
actual prompts are longer (the "Actual prompt tokens" column is the server's own count). The engine
serves one request at a time (`max-concurrency 1`); each point is a single request to an already
loaded engine, greedy sampling, 256 generated tokens. Memory peaks are the machine's use while the
request ran, not the engine's alone. Model load, seeding and warm-up are outside the measurements.

## Results

**v0.1.45-poly.7 (`fabae0d`), 2026-11-03**

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 63 | 0 | 111 | 1 | 79.5 | 57.9 | 0.826 |
| 256K | 266,673 | 0 | 256 | 1 | 4,217.3 | 77.8 | 63.4 |
| 512K | 534,873 | 0 | 256 | 2 | 3,863.3 (3,516.2–3,863.3) | 74.0 (70.9–74.0) | 138.7 (77.8–138.7) |
| 1M | 1,002,472 | 0 | 256 | 1 | 3,045.8 | 66.8 | 243.5 |

Memory at 1M: VRAM peak 31.69 GiB, RAM peak 121.51 GiB, expert cache hits 0.824, MTP acceptance 57.1%.
At 512K: 31.64 / 120.32 GiB, hits 0.807, acceptance 62.4%.

**v0.1.45-poly.6 (`20d839d`), 2026-10-31 — the same host and settings**

| Configuration | Actual prompt tokens | Generated tokens | Runs | Prompt tok/s | Decode tok/s | TTFT seconds | RAM peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short chat | 63 | 111 | 1 | 74.9 | 53.9 | 0.88 | 120.0 GiB |
| 256K | 266,673 | 256 | 1 | 4,128.3 | 70.0 | 64.7 | 121.0 GiB |
| 512K | 533,934 | 256 | 1 | 3,767.2 | 72.7 | 141.9 | 121.0 GiB |
| 1M | 1,000,964 | 256 | 1 | 3,230.1 | 62.6 | 310.3 | 121.0 GiB |

poly.7 against poly.6: short chat +6.1% prompt / +7.4% decode, 256K +2.2% / **+11.1%**, 512K +2.6% /
+1.8%, 1M −5.7% / +6.7%. RAM peak fell 5.3 GiB (121.0 → 115.7 GiB at 512K).

## Correctness and limitations

- One host, one model file, one power limit (450 W against the 575 W default) — prompt rates are not
  comparable with unlimited-power figures.
- One run per point except 512K. The 512K row shows both runs: the first landed at 3,516.2 / 70.9
  tok/s, the second at 3,863.3 / 74.0. Single runs cannot distinguish a build effect from run-to-run
  variance at this context length.
- The 1M point regressed on prompt rate (−5.7%) while decode improved (+6.7%); with one run each
  this is not established as a build effect.
- No output-quality or long-context retrieval checks were run; only speed, memory, expert-cache hit
  rate and MTP acceptance were recorded.
- Raw per-point JSON: `poly6-*.json` (measured 2026-10-31) and `poly7-*.json` (2026-11-03), one object
  per request, with the engine's own `prefill_tok_s` / `decode_tok_s` and the machine's VRAM/RAM peaks.
  Each file holds the run's short-chat row followed by the long-prompt row.
- **The first request after a model load reports a low prompt rate** — in `poly7-512k-run1.json` the
  short-chat row shows 5.6 tok/s while the long row in the same file is normal (3,516.2); the short
  numbers quoted above come from `poly7-short.json`, a separate invocation. Treat any single first
  prompt as unmeasured rather than as a build effect.
- `poly7-512k-run2.json` is the second run of the same point; its filename carries `run2`, not a power
  limit — both 512K runs were at the same 450 W.


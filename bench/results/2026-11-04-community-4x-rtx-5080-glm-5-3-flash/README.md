# Community benchmark on 4 × NVIDIA GeForce RTX 5080 (16 GB) — GLM 5.3 Flash, strata-poly

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run, beyond the five-probe sanity cell noted under *Checks*.

This answers the **"Three or four NVIDIA GPUs"** row of `docs/TEST_REQUESTS.md`, and adds the long soak from the row below it.

## Hardware and software

- **GPUs:** 4 × NVIDIA GeForce RTX 5080, 16 GiB each (64 GiB total), 360 W power limit each (default; not raised)
- **PCIe switch:** Broadcom / LSI **PEX89048 PCIe Gen 5, 48 port/lane** switch (`lspci` `1000:c034`, rev a0; upstream port at `02:00.0`, downstream ports at `03:00.0`, `03:04.0`, `04:00.0`)
- **Per-card link:** **PCIe Gen 5 x8** (measured under load: `pcie.link.gen.current 5`, `width.current 8` on all four; each card is x16-capable, wired x8 through the switch)
- **Topology** (`nvidia-smi topo -m`): GPU0–GPU1 **PIX**, GPU2–GPU3 **PIX**, all cross-pairs **PXB** — two adjacent-port pairs, every pair across the switch, no host bridge in any GPU-to-GPU path
- **P2P** (`nvidia-smi topo -p2p r`): **OK on all 12 ordered pairs**
- **CPU:** Intel Core Ultra 7 270K Plus, 24 cores / 24 threads, 1 socket, 1 NUMA node (no SMT siblings)
- **RAM:** 2 × 64 GB DDR5-6400 = 125.2 GiB usable
- **OS:** Linux 7.0.0-38-generic; GPU driver 610.57.04; **CUDA 13.3** (nvcc V13.3.73)
- **PolyStrata:** project 0.1.45, **commit `2d6a8f9b`** (main at time of measurement); engine 0.1.45, built from source with `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`
- **Background workloads:** none; the box is an inference host and nothing else ran during the suite

## Model and configuration

- **Model:** GLM-5.3-Flash, `GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn.gguf`, **113,585,695,168 bytes**; `setup.py --inspect` on this build reports architecture `glm5-next` and routed experts at **2.82 bits/weight (Q3 class)**
- **Draft block:** `draft-block.gguf` (MTP), 2,787,998,016 bytes, fetched and verified with `tools/glm_draft_fetch.py`
- **Context:** 32,768 tokens; parallel slots 1; layer split `auto` across the four cards
- **Coverage:** 4,652 of 12,384 routed experts in VRAM (38.0 GiB), identical on every arm — the pool is VRAM-limited, not profile-limited

```text
strata-poly --model GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf --max-context 32768 \
  --draft-model draft-block.gguf --expert-profile <arm>.profile --skip-weight 0.095
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server already running; model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file, 64 new tokens per turn, 3 sessions

## Results — `--skip-weight 0.095`

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 43 (40-48) | 0 | 100 (100-100) | 9 | not measured | 63.8 (50.7-72.1) | 0.70 (0.64-0.82) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 478.5 (478.2-479.0) | not measured | 8.46 (8.41-8.46) |
| 16K prompt, fresh | 16057 (16057-16057) | 0 (0-0) | 24 cap | 3 | 572.9 (572.7-573.1) | not measured | 28.13 (28.10-28.14) |
| agent session, follow-up turns | 14137 (14111-14137) at the last turn | see below | up to 64 | 3 | see below | 61.7 (52.2-67.5) | 7.50 (5.94-8.13) |

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.9 % (68.9-68.9); earlier-turn tokens read again per session: 5 (5-5); first turn (nothing cached) first token: 2.17 s (2.17-2.36).

## Side A/B — the flag is worth more than the build

Three boots of the same file, the same cards, the same context, each with its own copy of one frozen expert profile (the engine rewrites the profile after an answer, so a shared file would make every arm start from the previous arm's state):

| Arm | Decode tok/s (chat, median of 9) | Agent decode (median of 15) | 4K prompt tok/s | 16K prompt tok/s | Experts in VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| defaults | 48.8 (44.8-60.1) | 52.5 (42.1-60.0) | 509.5 | 592.1 | 4,652 / 12,384 |
| defaults (first boot, cold page cache) | 52.1 (45.9-57.5) | 52.3 (46.9-60.8) | 31.1 *(see note)* | 34.2 *(see note)* | 4,652 / 12,384 |
| **`--skip-weight 0.095`** | **63.8 (50.7-72.1)** | **61.7 (52.2-67.5)** | 478.5 | 572.9 | 4,652 / 12,384 |

**+22 % to +31 % decode** depending on which default boot it is compared against, at **no cost in coverage** and no measurable change in prefill. This reproduces, on current main, what we measured on the 2026-11-02 build (+24.3 % chat, +20.9 % agent over a four-boot interleaved A/B/A), so the flag's effect is stable across both builds. Chat decode has a ±7 % boot-to-boot band on this box — quote the range, not a single number.

Other flags measured and **not** recommended on a 16 GB card: `--prompt-batch 4096` (−16.8 coverage points, only +4 % prefill past the knee), `--threads 16` (−4 %, the engine already uses all 24), `--vram-reserve-mib 300` (+6.4 % but all four cards report `LOW` with ~210 MiB free).

## Side observation — the "fresh prompt" cells are not always fresh

On the first boot of this suite the 4K and 16K cells reported **4020 of 4021** and **16056 of 16057** prompt tokens **reused**, and prompt rates of 31.1 and 34.2 tok/s. With `--prompt-cache 0` the same cells read 0 reused at 509.5 and 592.1 tok/s. The default `--prompt-cache 6` can therefore serve a "fresh" benchmark prompt from checkpoints, which makes the prefill number in those cells depend on the boot rather than the engine. Worth knowing before comparing two machines' 4K/16K rows; our report quotes the `0 reused` cells.

## Long soak

One hour of back-to-back requests on the same boot and configuration (`--skip-weight 0.095`), no pauses, one client:

- **1,621 requests, 0 failures** in 60.0 minutes (five rotating prompts, greedy, 128 new tokens each; median wall 2.2 s per request)
- **decode p10 / median / p90 = 66.2 / 69.4 / 72.6 tok/s** (min 54.2, max 78.1)
- **no drift**: median decode by 15-minute quarter = 69.4 / 69.6 / 69.4 / 69.5 tok/s; draft acceptance 80.3 % in the first quarter against 80.6 % in the last; every quarter generated the same 51,840 tokens
- **thermals and memory**: GPU maximum 60 °C, VRAM steady at 15,634 MiB on all four cards, `MemAvailable` never below 118 GB, no OOM, no restart, no error of any kind

## What was not tested

- an unequal pair of cards (we have four of one model), Pascal, or a host with ≤32 GB of RAM — the other rows of `docs/TEST_REQUESTS.md`
- `parallel > 1` (one lane only), and any correctness benchmark beyond the five-probe cell noted above
- the 1M context: this file declares it, and a separate run read 986,960 tokens at 408.9 tok/s with the needle recovered, but that is outside this suite's cells
- concurrency, and any comparison against another engine


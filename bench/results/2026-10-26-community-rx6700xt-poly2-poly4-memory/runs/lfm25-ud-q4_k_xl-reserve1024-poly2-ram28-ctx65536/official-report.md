# Community benchmark on CPU only

Measured on 2026-10-26 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: lfm2.5-24b-a2b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model LFM2.5-24B-A2B-UD-Q4_K_XL.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib 0 --prompt-batch 2048 --kv-whole --threads 8 --vram-reserve-mib 1024
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a PolyStrata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 39 (36-45) | 0 | 100 (100-100) | 9 | not measured | 78.0 (69.3-87.2) | 0.21 (0.20-0.24) |
| 4K prompt, fresh | 4126 (4126-4126) | 0 (0-0) | 24 cap | 3 | 1523.0 (1492.2-1539.3) | not measured | 2.73 (2.70-2.78) |
| 16K prompt, fresh | 16490 (16490-16490) | 0 (0-0) | 24 cap | 3 | 1476.7 (1465.2-1478.0) | not measured | 11.20 (11.19-11.29) |
| 32K prompt, fresh | 32875 (32875-32875) | 0 (0-0) | 24 cap | 3 | 1118.0 (1111.2-1119.3) | not measured | 29.45 (29.42-29.63) |
| agent session, follow-up turns | 14448 (14448-14448) at the last turn | see below | up to 64 | 3 | see below | 77.0 (60.5-86.0) | 2.86 (1.99-3.23) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.8 % (67.8-67.8); earlier-turn tokens read again per session: 311 (311-312) (earlier turns that the cache did not hold; the 13766 (13765-13766) new tokens of the files are read once); first turn (nothing cached) first token: 0.72 s (0.70-0.79).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 563.5 | 0.705 | 59.7 |
| 1 | 3730 | 391 | 3339 | 1525.8 | 2.207 | 60.5 |
| 2 | 5402 | 3725 | 1677 | 845.1 | 1.99 | 68.4 |
| 3 | 8568 | 5397 | 3171 | 1084.0 | 2.94 | 77.0 |
| 4 | 11595 | 8563 | 3032 | 1077.3 | 2.831 | 75.7 |
| 5 | 14448 | 11590 | 2858 | 893.0 | 3.219 | 62.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 208 s.

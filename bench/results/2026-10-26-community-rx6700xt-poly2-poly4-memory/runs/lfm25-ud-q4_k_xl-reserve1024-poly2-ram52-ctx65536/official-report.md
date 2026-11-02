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
| short chat | 39 (36-45) | 0 | 100 (100-100) | 9 | not measured | 79.9 (70.8-87.0) | 0.22 (0.21-0.24) |
| 4K prompt, fresh | 4126 (4126-4126) | 0 (0-0) | 24 cap | 3 | 1545.3 (1513.0-1556.2) | not measured | 2.69 (2.67-2.75) |
| 16K prompt, fresh | 16490 (16490-16490) | 0 (0-0) | 24 cap | 3 | 1481.1 (1471.2-1484.8) | not measured | 11.16 (11.14-11.24) |
| 32K prompt, fresh | 32875 (32875-32875) | 0 (0-0) | 24 cap | 3 | 1119.5 (1112.2-1120.7) | not measured | 29.41 (29.38-29.61) |
| agent session, follow-up turns | 14428 (14404-14448) at the last turn | see below | up to 64 | 3 | see below | 72.1 (62.3-83.5) | 2.83 (1.99-3.19) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.8 % (67.8-67.9); earlier-turn tokens read again per session: 310 (301-310) (earlier turns that the cache did not hold; the 13756 (13723-13767) new tokens of the files are read once); first turn (nothing cached) first token: 0.68 s (0.64-0.75).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 619.7 | 0.641 | 58.8 |
| 1 | 3730 | 391 | 3339 | 1531.1 | 2.2 | 63.8 |
| 2 | 5393 | 3725 | 1668 | 840.4 | 1.991 | 69.9 |
| 3 | 8559 | 5388 | 3171 | 1089.0 | 2.92 | 74.2 |
| 4 | 11575 | 8554 | 3021 | 1077.8 | 2.812 | 70.4 |
| 5 | 14428 | 11570 | 2858 | 902.1 | 3.188 | 62.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 207 s.

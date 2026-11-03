# Community benchmark on CPU only

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: tiel-coder-35b-a3b-ud-q3_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Tiel-Coder-35B-A3B-MTP-UD-Q3_K_XL.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib 0 --kv-whole --threads 8 --vram-reserve-mib 1024
```

## Method

Run by `python tools/strata_bench.py --clients 2 --long` against the HTTP API of a PolyStrata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 224 (221-229) | 0 | 100 (100-100) | 9 | not measured | 68.7 (60.9-82.6) | 0.73 (0.70-0.78) |
| 4K prompt, fresh | 4025 (4025-4025) | 0 (0-0) | 24 cap | 3 | 1136.2 (1073.0-1138.6) | not measured | 3.55 (3.54-3.76) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1088.3 (1087.1-1089.2) | not measured | 14.77 (14.76-14.79) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 838.7 (836.1-841.7) | not measured | 38.24 (38.10-38.36) |
| agent session, follow-up turns | 15360 (15339-15365) at the last turn | see below | up to 64 | 3 | see below | 63.2 (51.9-72.0) | 4.13 (2.01-8.21) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.9 % (69.9-69.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14334 (14334-14334) new tokens of the files are read once); first turn (nothing cached) first token: 1.34 s (1.34-1.36).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 835 | 0 | 835 | 625.4 | 1.339 | 63.7 |
| 1 | 4403 | 844 | 3559 | 954.0 | 3.738 | 66.7 |
| 2 | 6041 | 4444 | 1597 | 788.3 | 2.033 | 56.1 |
| 3 | 9455 | 6083 | 3372 | 411.0 | 8.215 | 58.8 |
| 4 | 12420 | 9498 | 2922 | 695.9 | 4.209 | 62.0 |
| 5 | 15339 | 12450 | 2889 | 705.7 | 4.104 | 63.7 |

### Concurrency: 2 clients

Total 49.0 tok/s (46.2-49.4); first token p50 2.022 s, p95 3.578 s; request latency p50 4.052 s, p95 5.459 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 291 s.

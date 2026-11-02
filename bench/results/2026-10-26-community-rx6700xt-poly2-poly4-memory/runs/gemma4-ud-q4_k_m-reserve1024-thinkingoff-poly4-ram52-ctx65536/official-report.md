# Community benchmark on CPU only

Measured on 2026-10-26 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-ud-q4_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --max-context 65536 --expert-profile expert-profile.txt --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf --conversation-cache-mib 0 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 70.7 (61.8-89.7) | 0.59 (0.50-0.62) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 895.5 (893.4-896.6) | not measured | 4.59 (4.58-4.63) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 969.5 (964.8-971.2) | not measured | 16.91 (16.88-17.24) |
| 32K prompt, fresh | 32774 (32774-32774) | 0 (0-0) | 24 cap | 3 | 812.8 (810.0-814.8) | not measured | 40.34 (40.24-41.12) |
| agent session, follow-up turns | 15418 (15412-15440) at the last turn | see below | up to 64 | 3 | see below | 69.8 (58.3-79.7) | 4.82 (2.60-5.43) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.1 % (68.1-68.1); earlier-turn tokens read again per session: 251 (245-273) (earlier turns that the cache did not hold; the 14806 (14806-14806) new tokens of the files are read once); first turn (nothing cached) first token: 1.11 s (1.05-1.15).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 345.4 | 1.151 | 74.1 |
| 1 | 4162 | 389 | 3773 | 819.3 | 4.621 | 69.8 |
| 2 | 5870 | 4155 | 1715 | 656.7 | 2.625 | 69.2 |
| 3 | 9362 | 5863 | 3499 | 652.8 | 5.378 | 59.5 |
| 4 | 12374 | 9355 | 3019 | 615.5 | 4.926 | 72.3 |
| 5 | 15418 | 12367 | 3051 | 625.2 | 4.9 | 64.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 296 s.

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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 69.0 (55.7-80.3) | 0.58 (0.47-0.61) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 891.7 (888.9-893.8) | not measured | 4.60 (4.59-4.66) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 974.7 (967.8-975.1) | not measured | 16.82 (16.82-17.19) |
| 32K prompt, fresh | 32774 (32774-32774) | 0 (0-0) | 24 cap | 3 | 815.4 (810.9-816.6) | not measured | 40.21 (40.15-41.08) |
| agent session, follow-up turns | 15427 (15418-15434) at the last turn | see below | up to 64 | 3 | see below | 65.8 (55.9-76.8) | 4.78 (2.59-5.61) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.1 % (68.1-68.1); earlier-turn tokens read again per session: 260 (251-267) (earlier turns that the cache did not hold; the 14806 (14806-14806) new tokens of the files are read once); first turn (nothing cached) first token: 1.09 s (1.06-1.13).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 351.2 | 1.132 | 79.1 |
| 1 | 4162 | 389 | 3773 | 822.4 | 4.604 | 65.8 |
| 2 | 5870 | 4155 | 1715 | 664.2 | 2.595 | 72.1 |
| 3 | 9362 | 5863 | 3499 | 626.3 | 5.605 | 55.9 |
| 4 | 12373 | 9355 | 3018 | 635.6 | 4.77 | 71.6 |
| 5 | 15427 | 12366 | 3061 | 627.6 | 4.898 | 57.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 297 s.

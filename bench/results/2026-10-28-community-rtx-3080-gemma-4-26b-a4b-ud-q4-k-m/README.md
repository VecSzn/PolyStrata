# Community benchmark on NVIDIA GeForce RTX 3080

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3080, 10 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700KF (16 cores, 24 threads); installed RAM: 63.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 610.88; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-ud-q4_k_m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --max-context 32768 --expert-profile expert-profile.txt --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 87.1 (69.4-99.0) | 0.34 (0.31-0.36) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 1852.2 (1842.1-1855.7) | not measured | 2.23 (2.22-2.24) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 2698.4 (2677.4-2708.7) | not measured | 6.13 (6.06-6.28) |
| agent session, follow-up turns | 15247 (15244-15258) at the last turn | see below | up to 64 | 3 | see below | 79.9 (60.4-98.7) | 2.26 (1.79-2.54) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.8 % (67.8-67.8); earlier-turn tokens read again per session: 261 (258-272) (earlier turns that the cache did not hold; the 14625 (14625-14625) new tokens of the files are read once); first turn (nothing cached) first token: 0.66 s (0.65-0.73).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 543.3 | 0.733 | 79.6 |
| 1 | 3981 | 389 | 3592 | 1598.5 | 2.259 | 76.1 |
| 2 | 5689 | 3974 | 1715 | 954.6 | 1.806 | 84.4 |
| 3 | 9183 | 5682 | 3501 | 1454.1 | 2.421 | 72.8 |
| 4 | 12199 | 9176 | 3023 | 1393.4 | 2.184 | 81.8 |
| 5 | 15247 | 12192 | 3055 | 1255.5 | 2.447 | 68.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 91 s.

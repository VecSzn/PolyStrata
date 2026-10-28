# Community benchmark on NVIDIA GeForce RTX 3080

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3080, 10 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700KF (16 cores, 24 threads); installed RAM: 63.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 610.88; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-qat-ud-q4_k_xl
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --max-context 32768 --expert-profile expert-profile.txt --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf
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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 99.5 (77.7-112.6) | 0.34 (0.32-0.36) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 2035.4 (2033.6-2058.5) | not measured | 2.02 (2.02-2.02) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 2803.8 (2787.3-2805.9) | not measured | 5.89 (5.85-6.02) |
| agent session, follow-up turns | 15259 (15258-15265) at the last turn | see below | up to 64 | 3 | see below | 87.1 (58.2-107.9) | 2.28 (1.64-3.36) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.8 % (67.8-67.8); earlier-turn tokens read again per session: 273 (272-279) (earlier turns that the cache did not hold; the 14625 (14625-14625) new tokens of the files are read once); first turn (nothing cached) first token: 0.77 s (0.72-0.85).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 467.4 | 0.851 | 103.7 |
| 1 | 3981 | 389 | 3592 | 1740.3 | 2.076 | 91.2 |
| 2 | 5692 | 3974 | 1718 | 1052.3 | 1.642 | 88.4 |
| 3 | 9185 | 5685 | 3500 | 1557.4 | 2.278 | 84.9 |
| 4 | 12202 | 9178 | 3024 | 1520.7 | 2.004 | 94.6 |
| 5 | 15259 | 12195 | 3064 | 1338.2 | 2.303 | 87.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 90 s.

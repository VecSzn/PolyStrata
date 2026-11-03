# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-11-02 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 5700X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 591.86; CUDA: 13.1
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: deepseek-v4-flash-ud-iq2_xxs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf --max-context 32768 --expert-profile expert-profile.txt
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 3.4 (2.4-3.9) | 6.01 (4.87-6.58) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 104.2 (74.8-118.6) | not measured | 39.63 (34.87-55.19) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 129.4 (116.0-132.4) | not measured | 128.11 (125.14-142.84) |
| agent session, follow-up turns | 14280 (14276-14285) at the last turn | see below | up to 64 | 3 | see below | 1.7 (1.0-3.0) | 31.16 (26.91-56.94) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.1-69.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13442 (13442-13442) new tokens of the files are read once); first turn (nothing cached) first token: 19.17 s (18.46-24.93).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 611 | 0 | 611 | 32.0 | 19.173 | 3.0 |
| 1 | 3825 | 640 | 3185 | 103.6 | 30.804 | 1.0 |
| 2 | 5447 | 3867 | 1580 | 57.2 | 27.664 | 1.6 |
| 3 | 8570 | 5506 | 3064 | 98.5 | 31.159 | 1.1 |
| 4 | 11471 | 8609 | 2862 | 93.7 | 30.606 | 1.7 |
| 5 | 14280 | 11524 | 2756 | 57.2 | 48.242 | 2.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 2153 s.

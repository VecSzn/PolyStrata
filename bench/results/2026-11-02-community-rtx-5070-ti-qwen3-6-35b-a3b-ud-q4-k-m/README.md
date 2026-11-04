# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-11-02 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 5700X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 591.86; CUDA: 13.1
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib auto --tail-skip 6
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 163.8 (133.8-201.4) | 0.25 (0.23-0.29) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2158.0 (2132.9-2158.9) | not measured | 1.89 (1.89-1.90) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3104.9 (3103.7-3154.6) | not measured | 5.19 (5.12-5.20) |
| agent session, follow-up turns | 14994 (14987-14995) at the last turn | see below | up to 64 | 3 | see below | 152.5 (109.6-168.0) | 2.18 (1.71-2.51) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.6); earlier-turn tokens read again per session: 253 (246-254) (earlier turns that the cache did not hold; the 14167 (14167-14167) new tokens of the files are read once); first turn (nothing cached) first token: 0.90 s (0.89-0.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 658.1 | 0.944 | 160.0 |
| 1 | 4045 | 602 | 3443 | 1383.2 | 2.511 | 109.6 |
| 2 | 5680 | 4038 | 1642 | 851.5 | 1.956 | 129.0 |
| 3 | 9100 | 5673 | 3427 | 1510.3 | 2.292 | 134.1 |
| 4 | 12057 | 9093 | 2964 | 1377.9 | 2.171 | 152.5 |
| 5 | 14987 | 12050 | 2937 | 1346.4 | 2.202 | 151.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 74 s.

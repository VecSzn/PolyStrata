# Community benchmark on NVIDIA GeForce GTX 1650

Measured on 2026-10-30 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce GTX 1650, 4 GiB VRAM
- CPU: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz (4 cores, 8 threads); installed RAM: 23.9 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 621.01; CUDA: not read
- PolyStrata: project 0.1.45, commit 20d839d; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: clover-35b-a3b-preview-apex-i-nanoplus
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model Clover-35B-A3B-Preview.APEX-I-NanoPlus-V2.1.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib auto
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 11.6 (10.6-13.7) | 1.91 (1.72-2.14) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 135.7 (133.9-145.3) | not measured | 29.64 (27.67-30.04) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 110.1 (109.8-110.9) | not measured | 145.84 (144.83-146.25) |
| agent session, follow-up turns | 15056 (15049-15062) at the last turn | see below | up to 64 | 3 | see below | 9.4 (8.7-9.8) | 35.55 (17.15-39.24) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.1-69.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14238 (14238-14238) new tokens of the files are read once); first turn (nothing cached) first token: 7.91 s (7.82-8.34).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 73.2 | 8.339 | 10.8 |
| 1 | 4084 | 621 | 3463 | 104.6 | 33.165 | 9.8 |
| 2 | 5722 | 4125 | 1597 | 93.2 | 17.17 | 8.9 |
| 3 | 9140 | 5768 | 3372 | 87.6 | 38.543 | 9.2 |
| 4 | 12108 | 9186 | 2922 | 81.6 | 35.853 | 9.3 |
| 5 | 15049 | 12160 | 2889 | 73.7 | 39.239 | 9.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 1255 s.

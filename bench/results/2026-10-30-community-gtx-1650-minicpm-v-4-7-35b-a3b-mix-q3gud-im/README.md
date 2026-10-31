# Community benchmark on NVIDIA GeForce GTX 1650

Measured on 2026-10-30 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce GTX 1650, 4 GiB VRAM
- CPU: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz (4 cores, 8 threads); installed RAM: 23.9 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 621.01; CUDA: not read
- PolyStrata: project 0.1.45, commit 20d839d; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: minicpm-v-4.7-35b-a3b-mix_q3gud_im
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf --max-context 32768 --expert-profile expert-profile.txt
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 17.8 (15.6-20.6) | 1.32 (1.25-1.52) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 146.6 (143.1-154.0) | not measured | 27.44 (26.13-28.10) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 116.2 (115.7-117.3) | not measured | 138.27 (136.91-138.84) |
| agent session, follow-up turns | 15056 (15051-15061) at the last turn | see below | up to 64 | 3 | see below | 15.0 (14.2-15.6) | 33.96 (16.37-38.93) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.1-69.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14238 (14238-14238) new tokens of the files are read once); first turn (nothing cached) first token: 7.08 s (7.01-7.51).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 82.0 | 7.515 | 16.6 |
| 1 | 4082 | 619 | 3463 | 107.4 | 32.347 | 15.6 |
| 2 | 5722 | 4125 | 1597 | 97.8 | 16.368 | 14.2 |
| 3 | 9147 | 5775 | 3372 | 91.1 | 37.066 | 14.8 |
| 4 | 12114 | 9192 | 2922 | 86.2 | 33.956 | 15.0 |
| 5 | 15056 | 12167 | 2889 | 74.4 | 38.879 | 14.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 1251 s.

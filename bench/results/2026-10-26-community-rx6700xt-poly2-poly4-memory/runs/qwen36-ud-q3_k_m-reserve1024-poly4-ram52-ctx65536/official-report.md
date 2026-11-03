# Community benchmark on CPU only

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q3_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib 0 --tail-skip 6 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 76.0 (69.8-94.9) | 0.43 (0.38-0.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1073.8 (1071.6-1075.7) | not measured | 3.75 (3.74-3.76) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1080.2 (1069.7-1084.0) | not measured | 14.88 (14.83-15.03) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 850.5 (842.2-851.8) | not measured | 37.71 (37.65-38.08) |
| agent session, follow-up turns | 15128 (15126-15145) at the last turn | see below | up to 64 | 3 | see below | 69.6 (57.4-76.0) | 4.21 (2.30-5.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 240 (238-257) (earlier turns that the cache did not hold; the 14314 (14314-14314) new tokens of the files are read once); first turn (nothing cached) first token: 1.41 s (1.34-1.42).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 453.9 | 1.345 | 83.9 |
| 1 | 4197 | 602 | 3595 | 874.9 | 4.117 | 69.6 |
| 2 | 5831 | 4190 | 1641 | 714.6 | 2.303 | 75.3 |
| 3 | 9255 | 5824 | 3431 | 697.0 | 4.932 | 76.0 |
| 4 | 12216 | 9248 | 2968 | 726.1 | 4.098 | 74.2 |
| 5 | 15145 | 12209 | 2936 | 566.7 | 5.19 | 59.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 270 s.

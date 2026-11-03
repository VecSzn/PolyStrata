# Community benchmark on CPU only

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q3_k_m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib 0 --tail-skip 6 --kv-whole --threads 8
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 78.2 (64.4-97.5) | 0.39 (0.34-0.41) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1211.9 (1205.6-1216.4) | not measured | 3.32 (3.31-3.34) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1103.6 (1092.4-1104.2) | not measured | 14.56 (14.56-14.72) |
| agent session, follow-up turns | 15139 (15126-15147) at the last turn | see below | up to 64 | 3 | see below | 71.1 (59.9-74.4) | 3.73 (1.90-4.87) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 251 (238-259) (earlier turns that the cache did not hold; the 14314 (14314-14314) new tokens of the files are read once); first turn (nothing cached) first token: 1.08 s (1.08-1.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 562.6 | 1.086 | 71.4 |
| 1 | 4189 | 602 | 3587 | 989.5 | 3.632 | 72.3 |
| 2 | 5825 | 4182 | 1643 | 861.3 | 1.914 | 69.8 |
| 3 | 9245 | 5818 | 3427 | 775.2 | 4.43 | 71.7 |
| 4 | 12197 | 9238 | 2959 | 789.3 | 3.758 | 74.4 |
| 5 | 15126 | 12190 | 2936 | 609.1 | 4.829 | 64.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 145 s.

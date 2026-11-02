# Community benchmark on AMD Radeon

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 5600X 6-Core Processor (6 cores, 12 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit c54f0eaf; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: lfm2.5-24b-a2b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model LFM2.5-24B-A2B-UD-Q4_K_XL.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib 0 --vram-reserve-mib 1024 --prompt-batch 2048 --kv-whole
```

## Method

Run by `python tools/strata_bench.py --clients 2 --long` against the HTTP API of a PolyStrata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 39 (36-45) | 0 | 100 (100-100) | 9 | not measured | 86.4 (79.7-92.1) | 0.19 (0.18-0.21) |
| 4K prompt, fresh | 4126 (4126-4126) | 0 (0-0) | 24 cap | 3 | 1616.4 (1597.7-1618.2) | not measured | 2.57 (2.56-2.60) |
| 16K prompt, fresh | 16490 (16490-16490) | 0 (0-0) | 24 cap | 3 | 1407.0 (1391.5-1413.7) | not measured | 11.75 (11.69-11.88) |
| 32K prompt, fresh | 32875 (32875-32875) | 0 (0-0) | 24 cap | 3 | 1043.7 (1031.1-1044.0) | not measured | 31.54 (31.53-31.93) |
| agent session, follow-up turns | 14438 (14429-14448) at the last turn | see below | up to 64 | 2 | see below | 76.7 (68.5-83.9) | 2.71 (1.66-3.04) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.8 % (67.8-67.8); earlier-turn tokens read again per session: 310 (0-312) (earlier turns that the cache did not hold; the 13748 (0-13765) new tokens of the files are read once); first turn (nothing cached) first token: 0.54 s (0.54-0.54).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 758.3 | 0.538 | 67.5 |
| 1 | 3730 | 391 | 3339 | 1650.7 | 2.161 | 69.5 |
| 2 | 5402 | 3725 | 1677 | 1016.5 | 1.699 | 75.7 |
| 3 | 8568 | 5397 | 3171 | 1179.6 | 2.75 | 77.6 |
| 4 | 11595 | 8563 | 3032 | 1123.8 | 2.756 | 78.9 |
| 5 | 14448 | 11590 | 2858 | 949.7 | 3.039 | 68.5 |

### Concurrency: 2 clients

Total 77.7 tok/s (69.9-80.0); first token p50 1.074 s, p95 2.033 s; request latency p50 2.493 s, p95 3.57 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

### Skipped and failed

- failed agent: no token came back

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 206 s.

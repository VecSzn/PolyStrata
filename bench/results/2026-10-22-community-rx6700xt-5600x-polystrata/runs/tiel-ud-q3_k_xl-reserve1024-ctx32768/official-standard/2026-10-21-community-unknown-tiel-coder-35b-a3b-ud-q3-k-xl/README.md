# Community benchmark on AMD Radeon

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 5600X 6-Core Processor (6 cores, 12 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit c54f0eaf; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: tiel-coder-35b-a3b-ud-q3_k_xl
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model Tiel-Coder-35B-A3B-MTP-UD-Q3_K_XL.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib 0 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 224 (221-229) | 0 | 100 (100-100) | 9 | not measured | 73.1 (69.0-91.8) | 0.52 (0.50-0.59) |
| 4K prompt, fresh | 4025 (4025-4025) | 0 (0-0) | 24 cap | 3 | 1241.5 (1240.1-1241.5) | not measured | 3.25 (3.25-3.25) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1031.6 (1028.5-1036.6) | not measured | 15.58 (15.50-15.62) |
| agent session, follow-up turns | 15391 (15348-15396) at the last turn | see below | up to 64 | 3 | see below | 66.5 (57.3-77.6) | 4.11 (1.90-4.62) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (69.9-70.0); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14334 (14334-14335) new tokens of the files are read once); first turn (nothing cached) first token: 1.04 s (1.04-1.04).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 835 | 0 | 835 | 807.4 | 1.037 | 64.3 |
| 1 | 4403 | 844 | 3559 | 976.8 | 3.649 | 74.5 |
| 2 | 6041 | 4444 | 1597 | 843.9 | 1.897 | 68.4 |
| 3 | 9477 | 6104 | 3373 | 734.4 | 4.6 | 62.4 |
| 4 | 12455 | 9533 | 2922 | 712.2 | 4.111 | 69.4 |
| 5 | 15396 | 12507 | 2889 | 666.1 | 4.345 | 66.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 150 s.

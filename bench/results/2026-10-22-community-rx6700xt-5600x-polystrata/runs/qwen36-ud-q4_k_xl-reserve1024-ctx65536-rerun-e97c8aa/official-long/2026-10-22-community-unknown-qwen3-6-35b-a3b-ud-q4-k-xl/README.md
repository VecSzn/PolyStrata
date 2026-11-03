# Community benchmark on AMD Radeon

Measured on 2026-10-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 5600X 6-Core Processor (6 cores, 12 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit e97c8aa4; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 65536 --tail-skip 0 --kv-whole --conversation-cache-mib 0 --vram-reserve-mib 1024
```

## Method

Run by `python tools/strata_bench.py --clients 2 --long` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 84.5 (71.8-100.6) | 0.32 (0.31-0.34) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 998.2 (997.6-1001.9) | not measured | 4.03 (4.01-4.03) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 993.7 (992.3-993.9) | not measured | 16.17 (16.17-16.19) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 773.2 (771.4-773.4) | not measured | 41.47 (41.46-41.57) |
| agent session, follow-up turns | 15145 (15143-15160) at the last turn | see below | up to 64 | 3 | see below | 77.3 (66.6-81.1) | 4.31 (2.46-5.37) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 257 (255-272) (earlier turns that the cache did not hold; the 14314 (14314-14314) new tokens of the files are read once); first turn (nothing cached) first token: 1.16 s (1.16-1.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 513.7 | 1.189 | 95.4 |
| 1 | 4196 | 602 | 3594 | 843.5 | 4.267 | 77.3 |
| 2 | 5843 | 4189 | 1654 | 671.1 | 2.47 | 79.8 |
| 3 | 9273 | 5836 | 3437 | 675.5 | 5.095 | 73.5 |
| 4 | 12230 | 9266 | 2964 | 689.4 | 4.307 | 68.2 |
| 5 | 15160 | 12223 | 2937 | 548.7 | 5.361 | 80.3 |

### Concurrency: 2 clients

Total 65.0 tok/s (63.7-65.7); first token p50 1.314 s, p95 2.391 s; request latency p50 2.984 s, p95 3.998 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 298 s.

# Community benchmark on AMD Radeon

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 5600X 6-Core Processor (6 cores, 12 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit c54f0eaf; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 65536 --tail-skip 0 --kv-whole --conversation-cache-mib 0 --vram-reserve-mib 1024
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 81.4 (74.4-98.2) | 0.33 (0.31-0.34) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1000.9 (999.4-1002.7) | not measured | 4.02 (4.01-4.03) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 997.5 (994.2-998.5) | not measured | 16.11 (16.09-16.16) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 775.0 (772.7-776.5) | not measured | 41.38 (41.30-41.49) |
| agent session, follow-up turns | 15161 (15153-15162) at the last turn | see below | up to 64 | 3 | see below | 74.4 (66.1-85.0) | 4.33 (2.46-5.37) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 273 (265-274) (earlier turns that the cache did not hold; the 14314 (14314-14314) new tokens of the files are read once); first turn (nothing cached) first token: 1.19 s (1.15-1.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 491.7 | 1.242 | 90.4 |
| 1 | 4198 | 602 | 3596 | 832.0 | 4.328 | 75.4 |
| 2 | 5832 | 4191 | 1641 | 658.2 | 2.499 | 74.4 |
| 3 | 9257 | 5825 | 3432 | 671.8 | 5.117 | 68.2 |
| 4 | 12221 | 9250 | 2971 | 687.5 | 4.33 | 72.8 |
| 5 | 15153 | 12214 | 2939 | 549.6 | 5.355 | 66.1 |

### Concurrency: 2 clients

Total 63.2 tok/s (62.7-65.7); first token p50 1.412 s, p95 2.533 s; request latency p50 3.039 s, p95 4.076 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 304 s.

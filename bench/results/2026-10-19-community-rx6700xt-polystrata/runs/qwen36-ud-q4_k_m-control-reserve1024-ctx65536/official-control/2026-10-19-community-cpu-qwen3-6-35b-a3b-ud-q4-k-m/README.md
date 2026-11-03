# Community benchmark on CPU only

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --max-context 65536 --expert-profile expert-profile.txt --tail-skip 0 --kv-whole --conversation-cache-mib 0 --threads 8 --vram-reserve-mib 1024
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 79.7 (63.7-87.0) | 0.44 (0.42-0.49) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 912.1 (880.7-915.5) | not measured | 4.41 (4.39-4.57) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1021.3 (1017.4-1025.1) | not measured | 15.74 (15.68-15.80) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 816.6 (814.9-819.0) | not measured | 39.28 (39.16-39.35) |
| agent session, follow-up turns | 15128 (15115-15155) at the last turn | see below | up to 64 | 3 | see below | 76.5 (68.5-88.1) | 4.66 (2.77-5.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 240 (227-267) (earlier turns that the cache did not hold; the 14314 (14314-14314) new tokens of the files are read once); first turn (nothing cached) first token: 1.67 s (1.67-1.72).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 355.9 | 1.715 | 88.7 |
| 1 | 4185 | 602 | 3583 | 736.3 | 4.874 | 68.5 |
| 2 | 5817 | 4178 | 1639 | 592.4 | 2.774 | 75.7 |
| 3 | 9235 | 5810 | 3425 | 639.3 | 5.367 | 76.5 |
| 4 | 12187 | 9228 | 2959 | 660.9 | 4.487 | 74.2 |
| 5 | 15115 | 12180 | 2935 | 523.5 | 5.616 | 73.1 |

### Concurrency: 2 clients

Total 60.8 tok/s (58.3-61.0); first token p50 1.587 s, p95 2.764 s; request latency p50 3.245 s, p95 4.345 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 300 s.

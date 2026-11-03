# Community benchmark on CPU only

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: lfm2.5-24b-a2b-ud-q4_k_xl
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model LFM2.5-24B-A2B-UD-Q4_K_XL.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib 0 --prompt-batch 2048 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 39 (36-45) | 0 | 100 (100-100) | 9 | not measured | 84.0 (74.2-91.4) | 0.19 (0.17-0.20) |
| 4K prompt, fresh | 4126 (4126-4126) | 0 (0-0) | 24 cap | 3 | 1638.8 (1613.9-1664.2) | not measured | 2.54 (2.50-2.57) |
| 16K prompt, fresh | 16490 (16490-16490) | 0 (0-0) | 24 cap | 3 | 1501.2 (1484.8-1503.5) | not measured | 11.01 (11.00-11.13) |
| agent session, follow-up turns | 14430 (14430-14430) at the last turn | see below | up to 64 | 1 | see below | 72.6 (62.3-75.1) | 2.66 (1.82-3.06) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.8 % (67.8-67.8); earlier-turn tokens read again per session: 0 (0-311) (earlier turns that the cache did not hold; the 0 (0-13748) new tokens of the files are read once); first turn (nothing cached) first token: 0.62 s (0.62-0.62).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 637.6 | 0.623 | 64.3 |
| 1 | 3730 | 391 | 3339 | 1646.4 | 2.047 | 62.3 |
| 2 | 5384 | 3725 | 1659 | 914.8 | 1.819 | 74.7 |
| 3 | 8550 | 5379 | 3171 | 1148.4 | 2.775 | 72.6 |
| 4 | 11577 | 8545 | 3032 | 1147.1 | 2.66 | 75.1 |
| 5 | 14430 | 11572 | 2858 | 939.1 | 3.062 | 69.2 |

### Skipped and failed

- failed agent: no token came back
- failed agent: no token came back

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 79 s.

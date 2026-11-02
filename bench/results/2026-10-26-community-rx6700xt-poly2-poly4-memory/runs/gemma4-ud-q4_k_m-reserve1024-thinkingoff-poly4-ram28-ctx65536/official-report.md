# Community benchmark on CPU only

Measured on 2026-10-26 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-ud-q4_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --max-context 65536 --expert-profile expert-profile.txt --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf --conversation-cache-mib 0 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 75.2 (56.0-83.3) | 0.58 (0.50-0.62) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 891.9 (885.3-892.2) | not measured | 4.60 (4.60-4.68) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 970.2 (952.3-972.7) | not measured | 16.90 (16.86-17.47) |
| 32K prompt, fresh | 32774 (32774-32774) | 0 (0-0) | 24 cap | 3 | 812.5 (809.3-815.3) | not measured | 40.35 (40.21-41.15) |
| agent session, follow-up turns | 15414 (15411-15440) at the last turn | see below | up to 64 | 3 | see below | 66.5 (52.1-73.9) | 4.79 (2.57-5.44) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.1 % (68.1-68.1); earlier-turn tokens read again per session: 247 (244-273) (earlier turns that the cache did not hold; the 14806 (14806-14806) new tokens of the files are read once); first turn (nothing cached) first token: 1.08 s (1.06-1.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 333.5 | 1.192 | 74.5 |
| 1 | 4162 | 389 | 3773 | 818.4 | 4.626 | 66.8 |
| 2 | 5870 | 4155 | 1715 | 654.5 | 2.634 | 70.1 |
| 3 | 9365 | 5863 | 3502 | 646.4 | 5.437 | 58.9 |
| 4 | 12386 | 9358 | 3028 | 634.0 | 4.798 | 61.2 |
| 5 | 15440 | 12379 | 3061 | 623.5 | 4.93 | 57.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 297 s.

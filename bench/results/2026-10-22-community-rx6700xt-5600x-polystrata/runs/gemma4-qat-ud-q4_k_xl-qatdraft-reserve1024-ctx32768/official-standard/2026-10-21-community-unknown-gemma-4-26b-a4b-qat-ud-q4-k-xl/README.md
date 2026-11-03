# Community benchmark on AMD Radeon

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 5600X 6-Core Processor (6 cores, 12 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit c54f0eaf; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-qat-ud-q4_k_xl
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-poly --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --max-context 32768 --draft-model mtp-gemma-4-26B-A4B-it.gguf --conversation-cache-mib 0 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 87.7 (78.0-110.2) | 0.38 (0.38-0.50) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 1172.0 (1120.0-1181.5) | not measured | 3.51 (3.50-3.67) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 1009.3 (920.7-1011.6) | not measured | 16.42 (16.21-17.81) |
| agent session, follow-up turns | 15434 (15434-15437) at the last turn | see below | up to 64 | 3 | see below | 76.4 (54.1-81.5) | 4.29 (1.99-5.50) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.1 % (68.1-68.1); earlier-turn tokens read again per session: 267 (267-270) (earlier turns that the cache did not hold; the 14806 (14806-14806) new tokens of the files are read once); first turn (nothing cached) first token: 0.80 s (0.80-1.16).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 343.2 | 1.158 | 107.5 |
| 1 | 4162 | 389 | 3773 | 1002.8 | 3.777 | 80.1 |
| 2 | 5872 | 4155 | 1717 | 860.4 | 2.007 | 81.5 |
| 3 | 9365 | 5865 | 3500 | 738.4 | 4.756 | 61.8 |
| 4 | 12379 | 9358 | 3021 | 551.3 | 5.5 | 75.8 |
| 5 | 15434 | 12372 | 3062 | 687.3 | 4.473 | 75.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 154 s.

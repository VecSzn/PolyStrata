# Community benchmark on CPU only

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: tiel-coder-35b-a3b-ud-q3_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Tiel-Coder-35B-A3B-MTP-UD-Q3_K_XL.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib 0 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 224 (221-229) | 0 | 100 (100-100) | 9 | not measured | 66.2 (58.2-71.9) | 0.74 (0.68-0.79) |
| 4K prompt, fresh | 4025 (4025-4025) | 0 (0-0) | 24 cap | 3 | 1090.2 (1081.4-1092.6) | not measured | 3.70 (3.69-3.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1063.3 (1062.4-1073.8) | not measured | 15.12 (14.97-15.13) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 841.3 (836.9-841.9) | not measured | 38.12 (38.09-38.32) |
| agent session, follow-up turns | 15326 (15310-15351) at the last turn | see below | up to 64 | 3 | see below | 60.2 (53.5-69.8) | 4.30 (2.19-5.27) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.9 % (69.9-69.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14334 (14334-14334) new tokens of the files are read once); first turn (nothing cached) first token: 1.40 s (1.38-1.40).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 835 | 0 | 835 | 608.1 | 1.377 | 60.2 |
| 1 | 4403 | 844 | 3559 | 900.3 | 3.961 | 65.3 |
| 2 | 6040 | 4443 | 1597 | 730.1 | 2.194 | 53.5 |
| 3 | 9456 | 6084 | 3372 | 652.3 | 5.18 | 60.2 |
| 4 | 12408 | 9486 | 2922 | 664.7 | 4.406 | 65.6 |
| 5 | 15326 | 12437 | 2889 | 672.6 | 4.305 | 54.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 275 s.

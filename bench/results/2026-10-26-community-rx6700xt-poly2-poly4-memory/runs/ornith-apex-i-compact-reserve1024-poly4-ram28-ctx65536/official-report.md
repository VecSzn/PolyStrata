# Community benchmark on CPU only

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 5 3600 6-Core Processor (? cores, 9 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: ornith-1.5-35b-a3b-apex-i-compact
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib 0 --kv-whole --threads 8 --vram-reserve-mib 1024
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.2 (59.4-74.5) | 0.39 (0.36-0.41) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1018.8 (974.0-1023.6) | not measured | 3.95 (3.93-4.13) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 980.8 (971.5-985.4) | not measured | 16.39 (16.31-16.54) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 781.9 (781.0-784.5) | not measured | 41.02 (40.88-41.06) |
| agent session, follow-up turns | 15144 (15143-15191) at the last turn | see below | up to 64 | 3 | see below | 57.5 (54.2-67.6) | 4.33 (2.28-5.55) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 256 (255-303) (earlier turns that the cache did not hold; the 14314 (14314-14314) new tokens of the files are read once); first turn (nothing cached) first token: 1.23 s (1.21-1.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 498.4 | 1.225 | 59.6 |
| 1 | 4180 | 602 | 3578 | 840.3 | 4.265 | 56.6 |
| 2 | 5818 | 4173 | 1645 | 719.1 | 2.294 | 58.2 |
| 3 | 9238 | 5811 | 3427 | 664.8 | 5.164 | 60.7 |
| 4 | 12210 | 9231 | 2979 | 708.6 | 4.215 | 57.5 |
| 5 | 15143 | 12203 | 2940 | 538.4 | 5.47 | 57.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 293 s.

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
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model LFM2.5-24B-A2B-UD-Q4_K_XL.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib 0 --vram-reserve-mib 1024 --prompt-batch 2048 --kv-whole
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
| short chat | 39 (36-45) | 0 | 100 (100-100) | 9 | not measured | 88.5 (76.1-95.2) | 0.19 (0.17-0.22) |
| 4K prompt, fresh | 4126 (4126-4126) | 0 (0-0) | 24 cap | 3 | 1778.8 (1769.1-1782.8) | not measured | 2.33 (2.33-2.35) |
| 16K prompt, fresh | 16490 (16490-16490) | 0 (0-0) | 24 cap | 3 | 1414.0 (1406.5-1414.9) | not measured | 11.69 (11.68-11.75) |
| agent session, follow-up turns | not measured at the last turn | see below | up to 64 | 0 | see below | not measured | not measured |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: not measured; earlier-turn tokens read again per session: 0 (0-0) (earlier turns that the cache did not hold; the 0 (0-0) new tokens of the files are read once); first turn (nothing cached) first token: not measured.

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |

### Skipped and failed

- failed agent: no token came back
- failed agent: no token came back
- failed agent: no token came back

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 61 s.

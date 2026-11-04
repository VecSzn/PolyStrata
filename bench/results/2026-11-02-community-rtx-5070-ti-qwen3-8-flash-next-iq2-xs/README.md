# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-11-02 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 5700X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 591.86; CUDA: 13.1
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 32768 --kv int8 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 93.1 (82.0-112.1) | 0.30 (0.28-0.34) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1491.9 (1451.5-1497.1) | not measured | 2.71 (2.70-2.79) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2213.1 (2190.1-2215.2) | not measured | 7.28 (7.27-7.36) |
| agent session, follow-up turns | 15050 (15045-15070) at the last turn | see below | up to 64 | 3 | see below | 110.6 (88.1-139.8) | 2.57 (2.35-2.69) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (68.9-69.2); earlier-turn tokens read again per session: 4 (1-144) (earlier turns that the cache did not hold; the 14186 (14186-14187) new tokens of the files are read once); first turn (nothing cached) first token: 1.28 s (1.24-1.29).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 497.3 | 1.241 | 67.2 |
| 1 | 4030 | 620 | 3410 | 1278.4 | 2.687 | 88.1 |
| 2 | 5683 | 4088 | 1595 | 683.4 | 2.351 | 97.9 |
| 3 | 9113 | 5742 | 3371 | 1263.1 | 2.691 | 112.5 |
| 4 | 12092 | 9171 | 2921 | 1141.5 | 2.579 | 121.8 |
| 5 | 15045 | 12155 | 2890 | 1132.1 | 2.573 | 106.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 98 s.

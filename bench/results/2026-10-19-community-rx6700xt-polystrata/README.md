# RX 6700 XT 12 GB: official PolyStrata speed suite

Measured on 2026-10-19 from `26e4cb5df7996edfb5ff549a63b82d8581e1e658`, with the native HIP `gfx1031` engine 0.1.45. This report contains only the upstream `tools/strata_bench.py` suite. No custom quality tests are included.

## Hardware and environment

- GPU: AMD Radeon RX 6700 XT, 12 GB dedicated VRAM; one card used.
- CPU: AMD Ryzen 5 3600, 6 physical cores / 12 logical CPUs; eight engine workers.
- RAM: 32 GB installed. Unprivileged Debian 13 LXC limited to 28 GiB, no guest swap.
- Host kernel: 7.0.14-24-pve, existing amdgpu driver. No host driver change.
- HIP userspace: ROCm 7.13.0a20260515. Native gfx1031 source build; no architecture override.
- Storage: Intel SSDPEKNW010T8 NVMe SSD. Observed PCIe GPU link: 16.0 GT/s x16. The exposed power1_cap was 211 W at the metadata check; no power-limit change was made.
- Inference Docker: nonroot, offline, 26 GiB RAM/no swap, nine-CPU quota. Client: one CPU / 1.5 GiB.
- Other user LXCs were stopped. Model loading is outside the suite timings.

The benchmark client did not discover the AMD GPU through its hardware detector: generated directory names can say `cpu`, GPU fields can be empty, and RAM can reflect the host rather than the cgroup limit. These are retained as emitted. The native HIP build and external hardware identification above describe the actual device; no CPU-only execution is implied.

## Method and settings

Each context starts a fresh server, performs two short readiness requests, and then runs the upstream suite three times per test. Expert/cache adaptation within a running suite is retained. 32K uses the standard suite; 64K adds `--long --clients 2`. See every generated report for per-request timings, actual token counts, medians, ranges, agent-cache reuse and concurrency results.

With the configured server already running, the measured command is:

```sh
python tools/strata_bench.py --url http://127.0.0.1:8080 --config CONFIG.json --runs 3 --out OUTPUT
# For the 65536-token profile, also add: --long --clients 2
```

Each run has `settings.json` with the actual engine arguments and model hashes; `provenance.json` identifies the native build, installer revisions and available static tokenizer/pack hashes. Static expert-profile files were absent at post-run capture; their hashes are explicitly unavailable. Adapted output profiles are not substituted for missing initial hashes. Replace `<experiment>` with the installation directory. Conversation cache is disabled, the reasoning budget is 512 tokens, and explicit VRAM reserve is 1024 MiB for the main profiles. Upstream draft, tail-skip and batching defaults are retained unless the recorded arguments override them. `benchmark_mtp=0` is an adapter sentinel and does not disable engine speculation. Vision was disabled.

The Qwen UD-Q4_K_M control disables tail-skip and records its weight SHA comparison with a previous same-quant run. This is not an isolated engine comparison: runtimes, draft implementation and other defaults can differ. No historical throughput or private quality scores are included here.

## Successful suite summaries

Short-chat decode speed is not generation speed after a full context. Read the linked reports for sample counts and ranges.

| Profile | Short-chat median, tok/s | 16K prompt median, tok/s | Official report |
| --- | ---: | ---: | --- |
| qwen36-ud-q3_k_m-reserve1024-ctx32768 | 75.8 | 1104.0 | [report](runs/qwen36-ud-q3_k_m-reserve1024-ctx32768/official-standard/2026-10-19-community-cpu-qwen3-6-35b-a3b-ud-q3-k-m/README.md) |
| qwen36-ud-q3_k_m-reserve1024-ctx65536 | 79.9 | 1103.4 | [report](runs/qwen36-ud-q3_k_m-reserve1024-ctx65536/official-long/2026-10-19-community-cpu-qwen3-6-35b-a3b-ud-q3-k-m/README.md) |
| ornith-apex-i-compact-reserve1024-ctx32768 | 68.5 | 989.2 | [report](runs/ornith-apex-i-compact-reserve1024-ctx32768/official-standard/2026-10-19-community-cpu-ornith-1-5-35b-a3b-apex-i-compact/README.md) |
| ornith-apex-i-compact-reserve1024-ctx65536 | 69.2 | 991.5 | [report](runs/ornith-apex-i-compact-reserve1024-ctx65536/official-long/2026-10-19-community-cpu-ornith-1-5-35b-a3b-apex-i-compact/README.md) |
| tiel-ud-q3_k_xl-reserve1024-ctx32768 | 71.0 | 1077.7 | [report](runs/tiel-ud-q3_k_xl-reserve1024-ctx32768/official-standard/2026-10-19-community-cpu-tiel-coder-35b-a3b-ud-q3-k-xl/README.md) |
| tiel-ud-q3_k_xl-reserve1024-ctx65536 | 68.7 | 1088.3 | [report](runs/tiel-ud-q3_k_xl-reserve1024-ctx65536/official-long/2026-10-19-community-cpu-tiel-coder-35b-a3b-ud-q3-k-xl/README.md) |
| gemma4-ud-q4_k_m-reserve1024-ctx32768 | 73.5 | 988.3 | [report](runs/gemma4-ud-q4_k_m-reserve1024-ctx32768/official-standard/2026-10-19-community-cpu-gemma-4-26b-a4b-ud-q4-k-m/README.md) |
| gemma4-ud-q4_k_m-reserve1024-ctx65536 | 73.9 | 985.3 | [report](runs/gemma4-ud-q4_k_m-reserve1024-ctx65536/official-long/2026-10-19-community-cpu-gemma-4-26b-a4b-ud-q4-k-m/README.md) |
| minicpmv-mix_q3gud_im-reserve1024-ctx32768 | 59.2 | 1008.2 | [report](runs/minicpmv-mix_q3gud_im-reserve1024-ctx32768/official-standard/2026-10-19-community-cpu-minicpm-v-4-7-35b-a3b-mix-q3gud-im/README.md) |
| minicpmv-mix_q3gud_im-reserve1024-ctx65536 | 59.2 | 1006.1 | [report](runs/minicpmv-mix_q3gud_im-reserve1024-ctx65536/official-long/2026-10-19-community-cpu-minicpm-v-4-7-35b-a3b-mix-q3gud-im/README.md) |
| lfm25-ud-q4_k_xl-reserve1024-ctx32768 | 84.0 | 1501.2 | [report](runs/lfm25-ud-q4_k_xl-reserve1024-ctx32768/official-standard/2026-10-19-community-cpu-lfm2-5-24b-a2b-ud-q4-k-xl/README.md) |
| lfm25-ud-q4_k_xl-reserve1024-ctx65536 | 80.5 | 1490.1 | [report](runs/lfm25-ud-q4_k_xl-reserve1024-ctx65536/official-long/2026-10-19-community-cpu-lfm2-5-24b-a2b-ud-q4-k-xl/README.md) |
| qwen36-ud-q4_k_m-control-reserve1024-ctx32768 | 80.7 | 1019.4 | [report](runs/qwen36-ud-q4_k_m-control-reserve1024-ctx32768/official-control/2026-10-19-community-cpu-qwen3-6-35b-a3b-ud-q4-k-m/README.md) |
| qwen36-ud-q4_k_m-control-reserve1024-ctx65536 | 79.7 | 1021.3 | [report](runs/qwen36-ud-q4_k_m-control-reserve1024-ctx65536/official-control/2026-10-19-community-cpu-qwen3-6-35b-a3b-ud-q4-k-m/README.md) |
| qwen36-ud-q3_k_m-ctx32768 | 78.2 | 1103.6 | [report](runs/qwen36-ud-q3_k_m-ctx32768/official-standard/2026-10-19-community-cpu-qwen3-6-35b-a3b-ud-q3-k-m/README.md) |

## Failures and limits

The initial Qwen profile without an explicit reserve passed its 32K suite, retained separately as `qwen36-ud-q3_k_m`. Its 64K profile encountered an AMD memory-access fault during readiness, before the official suite started; there is no official 64K result for that attempt. Readiness and diagnostic outputs are not part of this official-results submission. The 1 GiB reserve was selected after a separate diagnostic, which is not presented as an independent official result. Inspect any included failed official stage receipts and client output alongside successful runs.

These measurements do not establish general model accuracy, production reliability, vision support or successful recall at the configured context limit. The upstream agent session is a speed/cache workload, not an external tool-execution accuracy test. No llama.cpp baseline was run for this report.

`index.json` enumerates included official stages and the measured tool hash. Copied upstream reports and per-run JSON are unchanged; settings metadata is separately labeled.

## Follow-up report

[October 25/26: v0.1.45-poly.2 versus v0.1.45-poly.4 at 28/52 GiB](../2026-10-26-community-rx6700xt-poly2-poly4-memory/README.md) adds a controlled memory-limit comparison on this card after the machine was upgraded to 64 GB. These original results remain the historical record; the new memory caps do not recreate the former physical DIMMs.

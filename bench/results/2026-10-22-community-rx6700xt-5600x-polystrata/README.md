# Community benchmark on AMD Radeon RX 6700 XT 12 GB (gfx1031), Ryzen 5 5600X

Measured 2026-10-20 to 2026-10-22 by mkw432. The official suite on four
models (Qwen3.6-35B UD-Q4_K_XL, LFM2.5-24B UD-Q4_K_XL, Tiel Coder
UD-Q3_K_XL, Gemma-4-26B-A4B QAT UD-Q4_K_XL) at 32K and 64K contexts, plus
a same-machine re-run of the Qwen suites at current main (e97c8aa) against
the original 0.1.45-c54f0eaf runs, a Gemma draft-pairing A/B, and a
separately labeled practical evaluation appendix. This is the second
report from an RX 6700 XT (after
[2026-10-19-community-rx6700xt-polystrata](../2026-10-19-community-rx6700xt-polystrata/README.md),
a Ryzen 5 3600 machine); the Qwen3.6 and LFM2.5 numbers reproduce that
report's rows within a few percent on different RAM and CPU.

Main limitation: single box, three days, one repetition of the optional
correctness screens (not three); the quality appendix is not a benchmark
suite and makes no claim beyond this hardware.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6700 XT 12 GB (gfx1031, PCI 1002:73DF,
  display-attached); 12272 MiB VRAM reported by amdgpu sysfs; PCIe gen4 x16
- CPU: AMD Ryzen 5 5600X (6 cores / 12 threads); installed RAM: 32 GB
  (31.2 GiB reported); storage: NVMe SSD (models on a second internal
  NVMe, ext4)
- OS: Ubuntu, Linux 7.0.0-38-generic; GPU driver: amdgpu (in-kernel);
  ROCm: AMD's Python wheels inside the project venv (no system rocm-smi)
- PolyStrata: release engine 0.1.45 built locally for gfx1031 (HIP); the
  Oct-20/21 runs used project commit c54f0eaf, the Oct-22 re-runs used
  e97c8aa (the engine binary is unchanged; the newer commits touch CUDA
  kernels and docs only)
- Background workloads: the Oct-20 LFM and Qwen suites ran while a large
  model download wrote to a second disk (labeled; see limitations); other
  runs idle-ish desktop; no power limits set

## Models and configurations

All GGUFs SHA-256 verified the same day (hashes in
[provenance.json](provenance.json)):

| Model | File | Repository |
| --- | --- | --- |
| Qwen3.6-35B-A3B UD-Q4_K_XL | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | unsloth/Qwen3.6-35B-A3B-MTP-GGUF |
| LFM2.5-24B-A2B UD-Q4_K_XL | `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` | unsloth/LFM2.5-24B-A2B-GGUF |
| Tiel Coder 35B-A3B MTP UD-Q3_K_XL | `Tiel-Coder-35B-A3B-MTP-UD-Q3_K_XL.gguf` | peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP |
| Gemma-4-26B-A4B it QAT UD-Q4_K_XL | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | unsloth/gemma-4-26B-A4B-it-GGUF |

Common launch settings (the community-parity profile): `--kv-whole`
`--conversation-cache-mib 0` `--vram-reserve-mib 1024`, `backend: hip` in
the run config. LFM adds `--prompt-batch 2048` and its setup-written
expert profile; Tiel adds `--threads 8` and its profile; Gemma adds
`--threads 8` and a draft model (see the A/B below); Qwen adds
`--tail-skip 0`. Contexts 32768 and 65536.

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 32768 --tail-skip 0 --kv-whole --conversation-cache-mib 0 --vram-reserve-mib 1024
```

## Method

`python tools/strata_bench.py --runs 3` through the HTTP API of a running
server (the Oct-22 64K re-run started the server itself; its
`engine-log-excerpt.txt` is included). Suite 1 as the guide describes:
thinking off, temperature 0, unique prompt tags, warm-up discarded, 3 runs
per test, medians and ranges. No other engine on the card during any run
(one engine at a time; the daily-driver server was stopped for the re-run
that started its own).

## Results

Full tables in each run folder's README; medians:

| Profile | Stage | Date / commit | Short chat decode | 4K prompt | 16K prompt | 32K prompt |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6 Q4_K_XL, ctx32768 | official-standard | 10-20 / c54f0eaf | 79.7 | 1053.1 | 999.7 | — |
| Qwen3.6 Q4_K_XL, ctx32768 | official-standard | 10-22 / e97c8aa | 82.2 | 1016.7 | 999.7 | — |
| Qwen3.6 Q4_K_XL, ctx65536 | official-long | 10-20 / c54f0eaf | 81.4 | 1000.9 | 997.5 | 775.0 |
| Qwen3.6 Q4_K_XL, ctx65536 | official-long | 10-22 / e97c8aa | 84.5 | 998.2 | 993.7 | 773.2 |
| LFM2.5 Q4_K_XL, ctx32768 | official-standard | 10-20 / c54f0eaf | 88.5 | 1778.8 | 1414.0 | — |
| LFM2.5 Q4_K_XL, ctx65536 | official-long | 10-20 / c54f0eaf | 86.4 | 1616.4 | 1407.0 | 1043.7 |
| Tiel Coder Q3_K_XL, ctx32768 | official-standard | 10-21 / c54f0eaf | 73.1 | 1241.5 | 1031.6 | — |
| Gemma-4 QAT Q4_K_XL, ctx32768, QAT-paired draft | official-standard | 10-21 / c54f0eaf | 87.7 | 1172.0 | 1009.3 | — |
| Gemma-4 QAT Q4_K_XL, ctx32768, repo Q8_0 draft | official-control | 10-21 / c54f0eaf | 81.4 | 1186.5 | 1024.3 | — |

All tok/s, engine-timed, medians of 3 runs (chat: median of 9 requests).
Agent-session stages and per-turn tables are in each run README; LFM's
32K agent stage did not complete (its Python-list tool-call format is
rejected by the stage, same gap as the 10-19 report).

Observations:

1. **Reproduction of the 10-19 RX 6700 XT report** (different CPU and
   RAM, same card): Qwen short-chat 79.7-84.5 vs their Q4_K_M control
   rows in the 80s class; LFM 86-89 matches the reported LFM2.5 row
   class; all four of our Qwen/LFM 32K and 64K rows land within a few
   percent of that report.
2. **Version delta on the same box, same prompts:** c54f0eaf → e97c8aa
   changed nothing measurable on HIP (32K prompt 999.7 = 999.7; 32K-row
   long 775.0 vs 773.2; chat deltas inside the run-to-run range).
3. **Gemma draft pairing:** swapping the QAT-paired 252 MB draft for the
   repo's Q8_0 draft (461 MB, the standard-model pairing) costs
   −7.2% short-chat decode (87.7 → 81.4) with prompt throughput
   unchanged (control ✓). Same model, same settings, only `--draft-model`
   differs. QAT users should keep the QAT-paired draft.
4. **Concurrent clients:** 2 clients total 65.0 tok/s on Qwen at ctx65536
   (below one client; `--batch` was not enabled on these servers).

## Correctness and limitations

See [PRACTICAL-EVALUATION.md](PRACTICAL-EVALUATION.md) for a separately
labeled, non-speed evaluation (blind agent screens and a day-desktop
soak) — it is not part of the official suite and no general accuracy
claim is made from it.

- The suite is a speed check; correctness is not measured by it.
- Oct-20 LFM and Qwen 32K suites ran while a large model download wrote
  to a second disk; prompt rates at 4K/16K match the idle re-run
  (1053.1/999.7 vs 1016.7/999.7), so the interference is not visible in
  the reported medians, but the condition is disclosed.
- Single machine; three days; day-to-day drift on this box measured at
  ~1-2% on a same-configuration llama.cpp control.
- GPU naming: this kernel's amdgpu sysfs exposes no `product_name`, so
  the server and `tools/strata_bench.py` (which now read sysfs) report
  the generic "AMD Radeon", and the report folder names say `unknown`
  (renamed to rx6700xt here; original generated names kept inside).
  The card is verified externally as an RX 6700 XT: PCI ID 1002:73DF,
  12272 MiB VRAM, gfx1031 engine build.
- No power limits, no clock capture; the card is display-attached.
- Failures: none in the official suites; one skipped test (the LFM 32K
  agent stage, above).

## Notes for RX 6700 XT / RDNA2 / 32 GB users

Collected during the same campaign; all on this hardware only:

- `--vram-reserve-mib 1024` matters: with the default reserve the expert
  cache grows into the last free MiB and a later request can fault in
  ROCm's buffer fill (the failure reported in #52's review thread). All
  rows here used an explicit reserve; zero memory faults across nine
  suite runs including 64K contexts.
- Qwen3.6 **Q6_K_XL is not usable on a 12/32 GB box**: native Strata
  cannot place it (32.6 GB file, placement timed out at >900 s even with
  all background load stripped — nominal fit is not placement slack),
  and strata-poly loads it (19 s, 2,344/10,496 experts resident) but
  faults with `illegal memory access` during a ~4.8K-token prefill.
  Short prompts generate fine (~25-37 tok/s). Q4_K_XL remains the pick.
- Qwen3.6 MTP lives **inside** the GGUF (~768 MB of the file); files
  from the plain `Qwen3.6-35B-A3B-GGUF` repo lack
  `nextn_predict_layers` and the engine refuses them — use the
  `Qwen3.6-35B-A3B-MTP-GGUF` repo.
- The web UI's GPU tiles need `"backend": "hip"` in the run config;
  setup-written configs lacked it and the tiles fell back to NVML
  ("not readable").
- No hipBLASLt tuning table ships for gfx1031; low-RAM-mode streaming
  models (IQ2/IQ1-class Flash-Next files) show prompt throughput 4-10x
  below the Q4-class rows here. `tools/hip/tune_hipblaslt` on this card
  is untested territory.
- Native Strata pins ~19-29 GB host RAM at load (32 GB boxes: bench it
  solo or watch the kernel OOM-killer); strata-poly's experts are
  page-cache based and tolerate background load — that difference is
  why every tabled number here is a poly number.

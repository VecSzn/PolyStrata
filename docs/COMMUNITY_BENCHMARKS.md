# Community benchmark results

Share measurements from your own hardware so other users can judge how Strata
might run on a similar machine. This guide suggests a report format and a pull
request workflow; a single model, GPU, or context length is enough to contribute.
The quickest way is the built-in benchmark: [`BENCH.bat` / `./bench.sh`](#run-the-benchmark) writes a report in this format.

Existing [benchmark reports](../bench/results/) provide examples. The
[engine 0.1.26 speed report](../bench/results/2026-09-29-speed-0126/README.md)
includes hardware, settings, and per-run JSON. The
[technical details](DETAILS.md#speed-measured) explain the published measurements
and their limits. Report what you actually measured and label estimates separately.

## The models PolyStrata adds

Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4 26B-A4B and GLM 5.3 Flash run on PolyStrata's own engine, `strata-poly`, so
their reports are listed here, apart from Strata's. The three below were posted on Reddit by people who ran the
engine of v0.1.41-poly.1 or an earlier one. The numbers are the posters' own, and "not stated" marks what a post
leaves out.

| Reported by | GPU | CPU, RAM, system | Model and file | Context | MTP | Writes, tokens/s | Reads, tokens/s | Before, or compared with |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [u/Present_Resource4022](https://www.reddit.com/r/LocalLLM/s/skrEbN9f48) | RTX 4070 SUPER 12 GB | CPU not stated; 32 GB DDR5-6000; Windows 11 | Qwen3.6-35B-A3B, HauhauCS uncensored Aggressive TTQ Q4_K_M | about 8K / about 32K | on | 140 / 100 | 2,800 to 3,300 | llama.cpp at 8K: about 77 writing and 2,600 reading, settings not stated |
| [u/hope_it_helps](https://www.reddit.com/r/LocalLLM/comments/1x22inc/comment/pf1lr00/) | RTX 4070 Ti 12 GB | not stated | Ornith 1.5 APEX-I-Compact | not stated | not stated | about 90, 60 at the slowest | not stated | usually 40 before, engine not stated |
| [u/Sensitive_Song4219](https://www.reddit.com/r/LocalLLM/comments/1x22inc/comment/pf3r111/) | RTX 5060 Laptop 8 GB | Core Ultra 9 275HX; 32 GB DDR5; system not stated | Tiel Coder 35B-A3B, file not stated | not stated | not stated | maybe 20% faster than before, no figure | 1,000 to 1,500; 1,016 at the slowest on a large read, over 1,400 in an auto-compact | 200 to 350 reading before, engine not stated |
| [iMilnb](https://github.com/VecSzn/PolyStrata/issues/13) | RX 7600 8 GB (gfx1102), the AMD beta | Ryzen 5 5600X; 30 GB; Linux, ROCm from AMD's wheels | Qwen3.6-35B-A3B UD-Q2_K_XL, and Ornith 1.5 APEX-I-Mini | 32K; the 24,000-token prompt is 31,834 tokens; `--kv-whole` | on | Qwen3.6: 74 in a short chat, 53 after the long prompt; Ornith: 40 and 43 | Qwen3.6 481, Ornith 444 | the first run of this engine on an AMD card, through the author's own port (#11) |
| [biurquezjoselina-design](https://github.com/VecSzn/PolyStrata/issues/10) | RTX 5090 D v2 24 GB | Core i7-14700KF, 13 threads; 128 GB DDR5; Windows 10 | GLM 5.3 Flash 3.0BIT (neuralll), 113.6 GB, with the draft block; 1,483 experts in VRAM | 32K, and 256K to 1M set in the run config | on | 26 to 30 after a short prompt, 23 to 30 after 2.5K to 10K | 213 to 395 at 2.5K to 10K; 488 at 112K down to 286 at 557K in one request each, every planted fact found; with 0.1.44.1-poly.5, 1,002,452 tokens in one request in 45.7 min (366 tok/s; 66.3 min with poly.2), the three facts found, and 17.7 tok/s writing after it | the first run of GLM on Windows; this project's 32 GB Linux pod reads 29K at 740 to 820 |
| [biurquezjoselina-design](https://github.com/VecSzn/PolyStrata/issues/21) | RTX 5090 D v2 24 GB | Core i7-14700KF, 13 threads; 128 GB DDR5; Windows 10 | DeepSeek V4 Flash UD-IQ2_XXS (Unsloth), 90.9 GB, on the release's ready-made engine | 32K, and 256K to 1M set in the run config | no MTP; guesses from repeated text | 35 to 36 after a short prompt, 29 to 34 after 2.5K to 10K | 303 to 469 at 2.5K, 688 to 805 at 10K; 393 at 252K, 247 at 504K and 138 at 1,008K in one request each, every planted fact found | the first run of DeepSeek V4 Flash on Windows; this project's 32 GB Linux pod read 487,529 tokens in 1,923 s |
| [biurquezjoselina-design](https://github.com/VecSzn/PolyStrata/issues/21) | RTX 5090 D v2 24 GB | Core i7-14700KF, 13 threads; 128 GB DDR5; Windows 10 | DeepSeek V4 Flash UD-IQ3_XXS (Unsloth), 104.2 GB, with 0.1.44.1-poly.5; 1,367 experts in VRAM | 32K, and 512K set in the run config | no MTP; guesses from repeated text | 27 to 31 after up to 10K tokens | 502 at 100K; 519,757 tokens in one request at 231 (705 at the start, 95 to 157 at the end), every planted fact found; RAM at most 119.8 of 127.8 GB | UD-IQ2_XXS on the same PC: 31 to 33 writing, 600 at 100K and 243 at 512K |
| [biurquezjoselina-design](https://github.com/VecSzn/PolyStrata/issues/21) | RTX 5090 D v2 24 GB | Core i7-14700KF, 13 threads; 128 GB DDR5; Windows 10 | DeepSeek V4 Flash UD-IQ4_XS (Unsloth), 136.7 GB, more than the RAM; 992 experts in VRAM | 32K, 256K set in the run config | no MTP; guesses from repeated text | with 0.1.44.1-poly.6: 3.4 after a short prompt, 2.2 to 2.8 after 2.5K to 10K; with poly.5: 1.0 to 2.1 | poly.6: 50 at 2.5K, 100 at 10K; poly.5: 23 and 53 | RAM at its 127.6 GiB ceiling and the rest paged from the disk; poly.6 loads in 28 s where poly.5 took 110 s and kept 58 GiB of converted MXFP4 experts in RAM; llama.cpp b11706, experts on the CPU, the same file and prompt: 8.6; with UD-Q4_K_XL (155 GB), llama.cpp 9.5 to 10.6 and poly.6 1.4 to 2.2 |
| [Haroldyolo](https://github.com/VecSzn/PolyStrata/discussions/47) | RTX 4070 Ti SUPER 16 GB | Ryzen 7 7800X3D; 30.9 GiB; Windows 11 | Qwen3.6-35B-A3B UD-Q3_K_M, 0.1.45-poly.1, `BENCH.bat` | 64K | on | 165 in a short chat (121 to 196), 143 in an agent session | 3,287 at 4K, 3,615 at 16K | the same PC runs Qwen3.8-Flash-Next Coder IQ1_M on the Strata project's engine at 59 writing, 2,033 and 2,646 reading |
| [Haroldyolo](https://github.com/VecSzn/PolyStrata/discussions/47) | RTX 4070 Ti SUPER 16 GB | Ryzen 7 7800X3D; 30.9 GiB; Windows 11 | Qwen3.6-35B-A3B UD-Q4_K_M, 0.1.45-poly.1, `BENCH.bat` | 128K | on | 164 in a short chat (119 to 212), 157 in an agent session | 2,043 at 4K, 3,078 at 16K | |
| [Haroldyolo](https://github.com/VecSzn/PolyStrata/discussions/56) | RTX 4070 Ti SUPER 16 GB | Ryzen 7 7800X3D; 30.9 GiB; Windows 11 | Tiel Coder 35B-A3B UD-Q4_K_XL, main at a418c11a, `tools/strata_bench.py` | 64K | on | 141 in a short chat (113 to 166), 118 in an agent session | 1,959 at 4K, 2,989 at 16K | the Qwen3.6 UD-Q4_K_M rows above, on the same PC |
| [Haroldyolo](https://github.com/VecSzn/PolyStrata/discussions/67) | RTX 4070 Ti SUPER 16 GB | Ryzen 7 7800X3D; 30.9 GiB; Windows 11 | MiniCPM-V 4.7 35B-A3B Q4_K_M, main at a418c11a, `tools/strata_bench.py` | 64K | no | 134 in a short chat (95 to 142), 123 in an agent session | 1,998 at 4K, 3,180 at 16K | the Tiel Coder and Qwen3.6 rows above, on the same PC |
| [Danielkgr](https://github.com/VecSzn/PolyStrata/pull/64#issuecomment-6376226685) | RX 7900 XTX 24 GB (gfx1100) and Radeon AI PRO R9700 32 GB (gfx1201, on a chipset x4 link), the layers split over both | Core i5-14600KF, 9 threads; 96 GB DDR4-3200; Linux, ROCm 7.14.1 from AMD's wheels | Ornith 1.5 122B Q5_K_M (ornith-ai), 82.6 GiB, main at 90347087; 6,213 of 12,544 experts in VRAM | 128K, `--kv-whole` | on, 65% to 85% of the guesses kept on short replies | 20.7 to 22.0 at 98K to 100K tokens; 28 to 31 in a short reply | 536 for a 98K-token prompt with `--prompt-batch 2048`, 472 with 1,024; later turns of the session 1.8 to 2.0 s | llama.cpp Vulkan, the same cards, file and session: 527 s for the first prompt where PolyStrata took 184, 6.9 s for later turns, 13.5 writing at 100K; against llama.cpp on the CPU, the same most likely token at 93% of 281 steps |
| [Dampfinchen](https://github.com/VecSzn/PolyStrata/discussions/44) | RTX 2060 | Core i7-9750H; 32 GB; system not stated | Gemma 4 26B-A4B QAT, file not stated; 555 experts in VRAM, `--vram-reserve-mib 120` | 64K, measured at 18K | not stated | 30 at 18K; slower as the context grows, about llama.cpp's speed near 85K | 900 at 18K | llama.cpp at 18K: 16 writing and 470 reading, steady as the context grows |
| [HaldunEymen](https://github.com/VecSzn/PolyStrata/issues/92) | RTX 3060 12 GB, PCIe 3.0 | Ryzen 5 5600G; 63 GB DDR4; Linux Mint 22.3, the models on a SATA hard disk | Tiel Coder 35B-A3B UD-Q3_K_XL and Ornith 1.5 APEX-I-Compact, main at 5e16c627, `polystrata.py benchmark --prompt-tokens 24000` | 32K; the long prompt is 32,201 tokens for Tiel, 32,022 for Ornith | on; Tiel 71% to 81% of the guesses kept, Ornith 74% to 95% | Tiel: 73 to 85 in a short chat, 52 to 60 after the long prompt; Ornith: 80 to 82 and 58 to 59 | Tiel 1,078, Ornith 1,141 to 1,166 for the long prompt; its first token after 79 s and 28 s | Qwen3.8-Flash-Next IQ3_XXS on the Strata project's engine, the same PC: 41 to 44 writing |
| [DRojasLey](https://github.com/VecSzn/PolyStrata/issues/98) | RX 7800 XT 16 GB (gfx1101) | Ryzen 7 5700X; 31.3 GB; Ubuntu 24.04.5, ROCm from AMD's wheels | Tiel Coder 35B-A3B UD-Q3_K_XL, main at 7b700758, `tools/strata_bench.py` | 128K, `--kv-whole`, `--kv q8_0`, `--vram-reserve-mib 4096` | on, 67% to 79% of the guesses kept | 86 in a short chat (80 to 102), 67 after a 32K prompt | 885 at 4K, 873 at 16K, 661 at 32K | the first report of Tiel Coder on an RX 7800 XT |
| [Csavoldi](https://github.com/VecSzn/PolyStrata/issues/88) | 2x Tesla V100-PCIE-32GB (sm_70), the layers split over both | Xeon E5-2682 v4; 47 GB; Linux | Qwen3.6-35B-A3B UD-Q3_K_M, main at 7b700758 built with CUDA 12.9.1 and `-DSTRATA_EXPERIMENTAL_SM60=ON`, `tools/strata_bench.py` | 32K | not stated | 157.5 in a short chat (135 to 196), 134.2 in an agent session | 2,615 at 4K, 2,778 at 16K | the first report of strata-poly on a Volta card |
| [GodrezJr2](https://github.com/VecSzn/PolyStrata/issues/107) | RTX 4060 Laptop 8 GB | Core i9-14900HX, 8 threads for the experts; 31.7 GB; Windows 11 | Qwen3.6-35B-A3B UD-Q3_K_M, 0.1.45-poly.4's ready-made engine, a 117,977-token prompt and 64-token answers | 128K, `--kv q8_0`, `--tail-skip 6` | not stated | 32.1 after the long prompt; 51.0 with `STRATA_FDEC=1` | 814; 1,318 with `STRATA_FATTN=1` | the two switches at 118K, which DETAILS.md measured only on an RTX 3070: 1.62x reading, 1.60x writing; both together 1,308 / 51.4 |
| [brocasm](https://github.com/VecSzn/PolyStrata/issues/115) | RX 7900 XTX 24 GB (gfx1100) | Ryzen 7 5700X; 31.2 GiB; CachyOS Linux, the HIP backend | Clover-35B-A3B-Preview Q4_K_M, 0.1.45-poly.6 (20d839db), `tools/strata_bench.py` | 128K, `--kv-whole` | not stated | 123 in a short chat (116 to 126), 112 in an agent session | 3,990 at 4K, 3,021 at 16K | |
| [LionelColaso](https://github.com/VecSzn/PolyStrata/pull/111) | GTX 1650 4 GB | Core i5-9300H; 23.9 GiB; Windows 11 | Clover-35B-A3B-Preview APEX-I-NanoPlus, 0.1.45-poly.6 (20d839d), `tools/strata_bench.py` | 32K | 13 of 33 guesses kept | 11.6 in a short chat (10.6 to 13.7), 9.4 in an agent session | 136 at 4K, 110 at 16K | |
| [LionelColaso](https://github.com/VecSzn/PolyStrata/pull/111) | GTX 1650 4 GB | Core i5-9300H; 23.9 GiB; Windows 11 | MiniCPM-V 4.7 35B-A3B mix_q3gud_im, 0.1.45-poly.6 (20d839d), `tools/strata_bench.py` | 32K | 26 of 60 guesses kept | 17.8 in a short chat (15.6 to 20.6), 15.0 in an agent session | 147 at 4K, 116 at 16K | the Clover row above, on the same PC |
| [HexhamAllstar](https://github.com/VecSzn/PolyStrata/pull/113) | RTX 3080 10 GB | Ryzen 7 5700X3D; 63.9 GiB; Windows 11 | Tiel Coder 35B-A3B UD-Q4_K_XL, 0.1.45, `tools/strata_bench.py` | 64K | on, 73% of the guesses kept | 77 in a short chat (66 to 85), 80 in an agent session | 1,420 at 4K, 2,527 at 16K | |

Setup offers Tiel Coder now (`--family tiel`), and not the HauhauCS file. How to give setup such a file is in the
[FAQ](FAQ.md#can-i-run-a-fine-tune-or-another-model-with-the-same-architecture).

To add yours, open an issue at [VecSzn/PolyStrata](https://github.com/VecSzn/PolyStrata/issues) with what
[What to record](#what-to-record) lists. `python polystrata.py benchmark --model <name> --prompt-tokens 24000`
gives the two speeds the README uses ([what it sends and prints](BACKENDS.md#benchmark)).

An upstream-suite report from an [RX 6700 XT 12 GB, Ryzen 5 3600 and 32 GB RAM](../bench/results/2026-10-19-community-rx6700xt-polystrata/README.md)
includes Qwen3.6, Ornith, Tiel Coder, Gemma 4, MiniCPM-V and LFM2.5 at 32K and 64K,
with per-request data, exact settings and a separate Qwen UD-Q4_K_M speed control. A second one from an
[RX 6700 XT 12 GB, Ryzen 5 5600X and 32 GB RAM](../bench/results/2026-10-22-community-rx6700xt-5600x-polystrata/README.md)
has Qwen3.6, LFM2.5, Tiel Coder and Gemma 4 QAT at 32K and 64K, a rerun of Qwen on a newer commit and a Gemma
draft A/B; its Qwen3.6 and LFM2.5 rows land within a few percent of the first.

An official-suite report from an [RTX 3060 12 GB, Xeon W-2145 and 32 GB RAM](../bench/results/2026-10-24-community-rtx3060-polystrata-poly2-poly3/README.md)
compares **v0.1.45-poly.2 and v0.1.45-poly.3** on Windows: five model configurations at 32K, a Qwen MMA pair,
and Qwen 64K/128K allocations with about 32K actual input. It includes thirteen suites with per-request data,
exact release commits and binary hashes, two retained LFM request failures, and an older-driver appendix.
These October 22/24 results do not test v0.1.45-poly.4 or later releases.

A [follow-up on the RX 6700 XT / Ryzen 5 3600, now with 64 GB RAM](../bench/results/2026-10-26-community-rx6700xt-poly2-poly4-memory/README.md)
compares **v0.1.45-poly.2 and v0.1.45-poly.4** at 28 and 52 GiB container memory limits, all at 64K configured context.
It includes 28 official suites, frozen expert-profile seeds, exact native HIP build hashes and four retained LFM
agent-request failures. The earlier report stays intact; the new limits compare available capacity on the upgraded
machine rather than recreating its old physical RAM.

The official suite on a [GTX 1650 4 GB with Clover APEX-I-NanoPlus](../bench/results/2026-10-30-community-gtx-1650-clover-35b-a3b-preview-apex-i-nanoplus/README.md)
and [with MiniCPM-V mix_q3gud_im](../bench/results/2026-10-30-community-gtx-1650-minicpm-v-4-7-35b-a3b-mix-q3gud-im/README.md),
and on an [RTX 3080 10 GB with Tiel Coder UD-Q4_K_XL](../bench/results/2026-10-30-community-rtx-3080-tiel-coder-35b-a3b-ud-q4-k-xl/README.md),
all three with v0.1.45-poly.6 and per-request data; their rows are in the table above.

### Comparing with llama.cpp

`bench/bench_vs_llama.py` is the script Strata used for
[its own comparison](../bench/results/2026-10-06-llamacpp-0.6.0/README.md). It starts PolyStrata and llama.cpp's
`llama-server` in turns on the same GGUF, one engine on the card at a time, sends both the same prompts and prints
the medians. With a size that setup has installed:

1. Stop PolyStrata and anything else that uses the card.
2. Download `llama-server` from llama.cpp's [releases](https://github.com/ggml-org/llama.cpp/releases), the CUDA
   build for an NVIDIA card. Qwen3.6's draft block needs `--spec-type draft-mtp`, which b11540 has.
3. Make a folder for the run and copy `prompt-4k.txt` and `prompt-32k.txt` into it from
   `bench/results/2026-10-06-llamacpp-0.6.0/`: the first 4,000 and 32,000 tokens of Strata's docs.
4. Save a file like this one as `machine.json` in the PolyStrata folder, with your paths: the size's GGUF,
   `llama-server`, the run config setup wrote for the size (`strata-<size>.json` in the PolyStrata folder), the
   PolyStrata folder's `serve` folder and the PolyStrata folder.

   ```json
   {
    "gguf": "D:/PolyStrata-data/models/qwen36-UD-Q3_K_M/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf",
    "llama_server": "D:/llama.cpp/llama-server.exe",
    "strata_cfg": "D:/PolyStrata/strata-qwen36-ud-q3_k_m.json",
    "serve": "D:/PolyStrata/serve",
    "cwd": "D:/PolyStrata",
    "configs": [
     {"name": "polystrata", "kind": "strata", "args": [], "override": {"--max-context": "36864"}},
     {"name": "llama-mtp", "kind": "llama", "args": ["--fit", "on", "-c", "36864", "-fa", "on",
      "-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]},
     {"name": "llama", "kind": "llama", "args": ["--fit", "on", "-c", "36864", "-fa", "on",
      "-ctk", "q8_0", "-ctv", "q8_0"]}
    ]
   }
   ```

5. Run it from the PolyStrata folder with setup's Python (`.venv/bin/python` on Linux), then print the table:

   ```
   .venv\Scripts\python.exe bench\bench_vs_llama.py run --machine machine.json --out <run folder> --rounds 3
   .venv\Scripts\python.exe bench\bench_vs_llama.py summary --out <run folder>
   ```

A round starts one engine, warms it up with two requests, sends the 4K prompt twice and the 32K prompt once at
temperature 0 and again at 0.7, each for up to 256 tokens of answer, and stops it; then the next engine starts.
Rounds alternate the order of the engines. The summary gives medians: prompt and answer speed, the server's prompt
time, the share of draft tokens accepted and the peak VRAM. Every request, and the command that started each
engine, is in `<run folder>/raw/`, and the engines' logs are in the run folder.

`--fit on` lets llama.cpp decide how many layers' experts go to the processor. A number picked by hand with
`-ngl 99 --n-cpu-moe N` can be faster; on the README's 8 GB laptop it was N = 30 for Qwen3.6 UD-Q3_K_M. The
context of 36,864 on both sides leaves room for the 32K prompt and its answer, and `-fa on` with the q8_0 cache is
how Strata's comparison ran llama.cpp. The example is for Qwen3.6 UD-Q3_K_M.

The README's laptop ran it on 2026-10-11 with the engine of 2ba29a383baf and llama.cpp b11540, for
[Qwen3.6 UD-Q3_K_M](../bench/results/2026-10-11-laptop-qwen36-llamacpp-b11540/README.md) and
[Tiel Coder UD-Q3_K_XL](../bench/results/2026-10-11-laptop-tiel-llamacpp-b11540/README.md). Against llama.cpp's
fastest setting in each cell, PolyStrata read the prompts 2.7 to 3.5 times as fast and wrote 1.5 to 1.8 times as
fast.

## Community reports

- [2026-09-30: RTX 5090, Core Ultra 9 285K, 64 GB RAM](../bench/results/2026-09-30-community-rtx-5090/README.md):
  Strata 0.1.29, original Flash-Next IQ2_XS, 131,072-token context; three runs
  each at 4,096, 32,768, and 128,000 prompt tokens, plus six recall checks.
- [2026-10-08: 2x TITAN RTX 24 GB (sm_75, NVLink present but unused), Xeon E5-2696 v4 without AVX-512, 125 GiB RAM](../bench/results/2026-10-08-community-2x-titan-rtx-0.1.40.3/README.md):
  Strata v0.1.40.3 source build, original Flash-Next IQ3_S, 262,144-token context (the model native window) with --vision and --reasoning-budget-tokens 12000; three runs each at 4,096, 32,768 and 128,000 prompt tokens. Decode 70.9/77.3/71.1 tok/s, prompt 868.9/1548.9/1626.9 tok/s at those three lengths, 8,649 cached experts, needles 6/6 at 32k and 128k. Notes that NVLink is not used by the engine.
- [2026-10-03: 2x AMD Instinct MI50 16 GB (gfx906), Xeon E5-2666 v3, 32 GB RAM](../bench/results/2026-10-03-community-2x-mi50/README.md):
  the gfx906 build (#638) with #639 and #640, Coder IQ1_M, 131,072-token context, layer split across both cards;
  three runs each at 4,096, 32,768, and 128,000 prompt tokens, plus six recall checks.
- [2026-10-04: 2x Intel Arc Pro B60 24 GB, Ryzen 5 5600, 64 GB RAM](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md):
  the SYCL port at `6f32ec0` with two fixes, Coder IQ1_M and Flash-Next IQ2_XS, 8,192-token context, layer split across both cards;
  short prompts and 2K-token prompts, plus one run on a single B60. Intel's SYCL engine, no 4,096/32,768/128,000-token sweep.
- [2026-10-07: 2x NVIDIA Quadro RTX 4000 8 GB, Xeon E5-2620 v3, 96 GB RAM](../bench/results/2026-10-07-community-2x-rtx-4000/README.md):
  Strata 0.1.38, Swift-Qwen3.8-Flash-Next IQ2_XS, 131,072-token context, layer split across dual GPUs with RAM-tiered expert cache;
  measured deep-context prompt ingestion (73k–74k tokens at 100–250 tok/s, 18–23 tok/s decode) and production pipeline comparison against a 12B model.
- [2026-10-08: Tesla V100-SXM2-32GB (sm_70), Threadripper 2990WX without AVX-512, 96 GB RAM](../bench/results/2026-10-08-community-v100-sm70-table/README.md):
  Strata `fb58e0d` (v0.1.41) source build for sm_70, Unsloth UD-IQ4_XS with a native-experts pack,
  262,144-token context, `--kv int8`, `--prefill auto`; **confirms the opt-in Volta decode kernels**
  (`STRATA_SM70_TABLE=1`) that [NVIDIA_V100.md](NVIDIA_V100.md) asks a V100 owner to check: decode
  **+9.8% and +15.0%** in two order-swapped pairs (28.16/28.64 → 30.92/32.94 tok/s), prompt throughput
  unchanged, and the generated tokens **bit-identical** in all four runs. Also notes that
  `tools/ab_engine.py`'s three short chats are below this machine's noise floor (the long-prompt arm
  points the same way, +5%).
- [2026-10-08: one AMD Radeon AI PRO R9700 32 GB, two EPYC 9334 CPUs, 503 GiB RAM](../bench/results/2026-10-08-community-r9700-linux/README.md):
  Linux/ROCm 7.2.3, frozen experimental 0.1.40.3 source, original Flash-Next IQ3_S fresh 1K/4K/32K/128K
  at three repeats each; separate IQ3_S five-pair and IQ2_XS two-pair short-increment comparisons for #1107.
  Includes per-request data and regressions; no all-workload recommendation or newer-main measurement.
- [2026-10-09: Radeon 8060S / Ryzen AI Max+ 395 Strix Halo laptop (HP ZBook Ultra G1a, 70 W), 128 GB unified memory](../bench/results/2026-10-09-community-strix-halo-laptop-70w/README.md):
  Strata v0.1.41 source build, ROCm 7.14.1, Unsloth UD-IQ4_XS, `docs/STRIX_HALO.md` §5 fast config with `--spec 5 --mtp-window 8192`;
  4–6 fresh-process runs at 8,192 / 32,768 / 131,072 prompt tokens: prompt 969 / 1,067 / 1,031 tok/s, decode 59.2 / 57.8 / 42.0 tok/s
  (medians). Prompt is power-limited (70 W package limit); needle 15/15, agentic 5/5.
- [2026-10-09: RTX 3060 12 GB, Core i7-12700, 64 GB RAM](../bench/results/2026-10-09-community-rtx-3060-0141/README.md):
  Strata v0.1.41 official Windows release, original Flash-Next Q2_0, IQ2_XS, IQ3_XXS and IQ3_S; prompt sweep from 128 to 128K tokens (260K for three of the four), plus recall, HumanEval 0-39, cache, MTP and concurrency checks.
- [2026-10-10: AMD Radeon RX 9070 16 GB (gfx1201), Ryzen 7 5800X3D, 64 GB RAM](../bench/results/2026-10-10-community-rx-9070/README.md):
  Strata 0.1.41 prebuilt Windows HIP engine, Flash-Next IQ3_XXS, 131,072-token context; three runs each at 4,195,
  33,467 and about 130,000 prompt tokens with a 256-token cap, plus six recall checks. Decode 58.7/57.9/57.9 tok/s,
  prompt 586/804/833 tok/s at those three lengths, needles 6/6 at 33.5k and 130k; one engine session, no clock capture.
- [2026-10-10: AMD Radeon RX 9060 XT 16 GB (gfx1200), Ryzen 7 9800X3D, 64 GB RAM](../bench/results/2026-10-10-community-rx-9060-xt/README.md):
  Strata 0.1.40.3 prebuilt Windows HIP engine, Flash-Next IQ2_XS and IQ3_XXS: a prefill-size sweep (8,192 best), a matched 32K coding comparison (9/9 tasks each, IQ2_XS about 17% faster decode), one 113K recall request. Sanitized data, fixtures and scripts included.
- [2026-10-10: 2x Tesla P40 24 GB, Xeon E5-2699 v3, 125 GB RAM](../bench/results/2026-10-10-community-2x-p40-iq3_s/README.md):
  Strata 0.1.41 CUDA 12 source build (sm_61), Flash-Next IQ3_S, 131,072-token context, layer split; decode 35.0 tok/s, prompt 350/558 tok/s at 3.9K/21.9K, needles 6/6, conversation checkpoints.
- [2026-10-12: Tesla P40 24 GB (sm_61), 20 vCPUs of a Xeon E5-2695 v4 without AVX-512, 98 GB RAM](../bench/results/2026-10-12-p40-sm61-kernels/README.md):
  Strata `a0116b2` (0.1.43.1) CUDA 12.4 source build, Flash-Next IQ3_S, 262,144-token context, the profile of a prompt and of a decode window on Pascal and four bitwise changes for sm_61:
  a 565-token prompt 205 -> 255 tok/s, decode 34-35 -> 37-38 tok/s, and with the opt-in `STRATA_DENSE_F32=1` a 7K prompt 360-432 -> 464-474 tok/s. Short and 7K prompts only; cuBLAS 12.4 only.
- [2026-10-09: RTX 5060 Ti 16 GB + Xeon E5-2680 v4 (Windows 10)](../bench/results/2026-10-09-community-rtx-5060-ti-e5-2680-v4/README.md):
  Strata 0.1.41, Coder IQ1_M and abliterated Q2_0 at 262K; prompt and decode at 4K / 32K / 128K.
- [2026-10-09: RTX 5060 Ti 16 GB under WSL2, Ryzen 7 3700X, 128 GB RAM](../bench/results/2026-10-09-community-rtx-5060ti-wsl2-iq3_s-0141/README.md):
  Strata 0.1.41 source build, IQ3_S with experimental speed projection; single runs only (decode 48.2/47.7 tok/s, prompt 343/598 tok/s at 3.8K/19.2K), no needle test, no TTFT.
- [2026-10-09: Radeon Pro VII 16 GB (gfx906), Xeon E5-2666 v3, 125.7 GiB RAM](../bench/results/2026-10-09-community-radeon-pro-vii-16gb/README.md):
  Source builds of 0.1.40.4 and 0.1.41 on the distribution's ROCm 6.2 (host patches in BUILD.json), IQ2_XS at 131K with KV streaming; decode 24.2/24.0/23.7 tok/s, prompt 350/339/284 tok/s at 4K/32K/128K, two sweeps with telemetry. Build logs trimmed (TRIMMED.md).
- [2026-10-08: RTX 4070 12 GB, Core i5-12600KF, 32 GB DDR4](../bench/results/2026-10-08-community-rtx4070-iq2-xs/README.md):
  IQ2_XS with adaptive cache E4/S24: 48.23 accepted-decode tok/s (median of 5) on a modified, locally built engine (not an official release; source commit and hashes in provenance.json, steps in REPRODUCE.md).
- [2026-10-12: Radeon 780M (gfx1103), Ryzen 7 8745HS, 60 GB RAM, GTT 50 GiB](../bench/results/2026-10-12-community-gfx1103/README.md):
  the gfx1103 port (native dot4, the gfx11 WMMA kernels, the exact speed switches and a calibrated hipBLASLt table), hand-built on ROCm 10.1, Q2_0 + MTP at 32K: prompts 90 tok/s plain, 186 with the table, 277 with the WMMA opt-ins (4K), decode 13.1 -> 14.3 tok/s; zero GPU resets.
- [2026-10-26: 2x Tesla V100-PCIE-32GB (sm_70), Xeon E5-2682 v4, 47 GB RAM](https://github.com/VecSzn/PolyStrata/issues/95):
  PolyStrata 0.1.45 in Docker on a CUDA 12.9 image built for sm_70, Qwen3.8-Flash-Next UD-IQ4_XS (Unsloth), 262,144-token context, `--kv int8`,
  split over both cards; `tools/strata_bench.py --long`: chat 53.9 tok/s (27 to 89 as the expert cache warms), prompt
  1,242 / 1,885 / 2,244 tok/s at 4K / 16K / 32K, agent turns 62.7 tok/s at a 69% cache hit.

## Run the benchmark

The simplest report is one command after setup. Windows: double-click `BENCH.bat`. Linux: `./bench.sh`. Or
`python tools/strata_bench.py` from the Strata folder. Setup also offers it at the end of an interactive install
(default no).

It starts the model with the settings you already have (the newest `strata-*.json`; `--config FILE` picks another one),
or uses a Strata server that is already running on that port (`--url http://127.0.0.1:8080` for another address). It then
sends a fixed suite through the HTTP API, three runs each, and reports medians with ranges. The default takes about
3 to 10 minutes once the model has loaded:

| Test | What it sends | What it reports |
| --- | --- | --- |
| chat | three short questions, up to 100 new tokens | decode tok/s, time to first token |
| 4K and 16K prompt | a fresh word-list prompt of about 4,096 and 16,384 tokens (skipped when your context is smaller) | prompt tok/s, time to first token |
| agent session | a system prompt and four tools, then five turns that each add about 1,500 words of a project file and continue from the model's own earlier answers | tokens the conversation cache reused, tokens read again, time to first token per turn |
| `--clients N` | N requests at the same moment (the server needs `--batch` to run them together) | total tok/s, p50 and p95 first-token time and latency |
| `--long` | the prompt test at about 32,768 tokens | prompt tok/s, time to first token |

Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second` in the answer); first-token time and
latency are measured by the client. Thinking is off for the suite so the answers are short and the runs comparable
(`--think` leaves it on). Every prompt carries a unique tag, so the fresh-prompt tests never get help from the cache.
Close other programs first: a browser or a game changes the numbers.

The result goes to `bench/results/<date>-community-<gpu>-<model>/` (or `--out DIR`):

- `README.md` in the format below, with the GPU, VRAM, CPU, RAM, OS, driver, Strata and engine versions, model and
  quantization, context and your settings filled in from the machine and the server
- `results.json` with every request (tokens, timings, errors) and the machine facts
- `engine-log-excerpt.txt` with the engine's start-up lines (when the benchmark started the server)

Home folders, user and host names, local network addresses and API keys are replaced by placeholders before anything is
written. Read the report once before you share it: the benchmark cannot know what else you consider private.

To share it, open a pull request that adds the folder (see [Submit a report](#submit-a-report)), or paste
`README.md` into a GitHub issue. Add what the tool does not know: background load, power limits, storage.

To compare with an earlier run on the same machine (before and after an update or a setting change):

```bash
python tools/strata_bench.py --compare bench/results/2026-10-01-community-rtx-3060-qwen3-8-flash-next
python tools/strata_bench.py --compare OLD_FOLDER --compare-only NEW_FOLDER   # two saved runs, nothing is started
```

The tool is a speed check, not a quality test. For recall and other checks, see [Optional correctness
checks](#optional-correctness-checks); for A/B tests of two builds use `tools/ab_engine.py`.

## What to record

Include enough information for someone else to repeat your run:

- **Hardware:** GPU model, VRAM, selected GPUs for a multi-GPU run, CPU, installed
  RAM, and storage type. Include PCIe link speed and width, GPU power limits, and
  other workloads when known or relevant.
- **Software:** OS, Strata commit and engine version, driver and CUDA version
  (or ROCm for HIP), and release binary or source build. Include changed build options.
- **Model:** exact repository and revision when available, quantization, GGUF
  filenames, and vision encoder if enabled. Identify custom packs, expert
  profiles, or draft vocabularies, with hashes or reproducible preparation steps.
- **Settings:** launch command and relevant run configuration, context limit, KV
  type and streaming window, expert cache, low-RAM mode, prefill size, CPU workers,
  MTP, reasoning effort, sampling, and vision. State whether calibration or
  experimental speed projection was enabled; attach the resulting settings.
- **Workload:** benchmark command or script, shareable prompts or instructions to
  build them, actual prompt and generated token counts, output cap, repetitions,
  and cache state for each run.

Use GB or GiB consistently and name the unit. Mark unavailable values explicitly
instead of guessing. Remove credentials and private content from configurations,
prompts, and logs before publishing them.

## How to measure

Start with a configuration that already works on your machine. You do not need to
download every model or test the largest context. A useful initial speed report
tests one short prompt and one longer prompt that fits, with a fixed output cap
(for example, 256 tokens). Record the actual output length if the model ends early.

For each configuration, try at least three measured runs and publish all of them
with a median and range. If you only have one run, say so. Explain any warm-up and
whether model loading is included in the timing.

Strata reuses conversation prefixes and adapts its expert cache. Separate prompts
that are fully processed from follow-ups that reuse tokens, and report the number
of reused and freshly read tokens from the engine log. Describe how you reset or
retained state between runs. A warmed expert cache and a reused prompt prefix are
different conditions; record both when possible.

Keep these measurements separate:

- **Prompt throughput:** freshly processed prompt tokens per second, from the
  engine's timing output. Keep the corresponding token count and duration.
- **Decode throughput:** generated tokens per second, from the engine's timing
  output. Count reasoning tokens too, or explain any different convention.
- **Time to first token:** seconds from sending the request to its first generated
  token. For a streaming API measurement, ignore keep-alives and empty deltas and
  state whether the first token is reasoning or answer text.
- **Total latency:** elapsed seconds for the whole request. State whether measured
  at the engine or client, and whether loading, vision encoding, or queueing is included.
- **Memory:** observed RAM and VRAM usage, including whether the value is a startup
  snapshot or a peak during inference. Report paging or out-of-memory failures.

Attach the engine timing lines and your measurement script if you used one. Avoid
calculating decode throughput by dividing generated tokens by total request time,
which also includes prompt processing. Cancelled or failed requests belong in the
report as failures, outside the successful throughput summary.

When comparing versions or settings, run both on the same machine and workload
and list every changed setting. Preserve the per-run results: output text and
draft acceptance can change speed even when the input is identical.

## Optional correctness checks

Speed measurements alone do not establish answer quality. Include a small check
relevant to your workload, such as a completed coding task with test results, a
tool call followed by its result, or an image question with an expected answer.

For long-context recall, the repository includes
[`tools/needle_bench.py`](../tools/needle_bench.py). With a running local server,
from the repository root:

```bash
python tools/needle_bench.py --help
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

Choose lengths your configured context can hold. The script supports `--api-key`
for an authenticated server; keep that key out of published commands. Save the
reported actual prompt lengths, misses, errors, and skipped cases as well as
successes. A needle test measures recall on those inputs, not overall model quality.

## Submit a report

1. Fork Strata and create a branch for your report.
2. Add a folder such as `bench/results/YYYY-MM-DD-community-rtx-5090/`. Use the
   measurement date and a short hardware label; add a suffix if the name exists.
3. Put a `README.md` based on the template below in that folder, together with
   per-run JSON or CSV and small scripts or prompts needed to repeat it. Use your
   own data format and document its fields and units. Keep large model files and
   generated packs out of the PR.
4. Open a pull request against `Niko1221/Strata:main`. Summarize the hardware,
   model, configurations tested, and limitations. Keep a results-only submission
   separate from engine changes so reviewers can assess the measurements directly.

Maintainers can review the report and decide where to include it. Adding a report
does not require changing the README's headline performance claims. For a failed
run or a suspected bug, a GitHub issue with the configuration and relevant logs
may be more useful than a benchmark PR.

## Report template

Copy this into your report's `README.md`, replace the placeholders, and remove
sections that do not apply. Use `not measured` for missing measurements; the table
contains no example performance numbers.

````markdown
# Community benchmark on GPU name

Measured on DATE by HANDLE. State what was tested and the main limitation.

## Hardware and software

- GPU and VRAM; CPU; installed RAM; storage; PCIe link if known:
- OS; driver; CUDA or ROCm:
- Strata commit; engine version; release binary or source build:
- Background workloads and any power limits:

## Model and configuration

- Model repository and revision; quantization; GGUF filenames:
- Vision encoder; custom packs or profiles:
- Context; KV type and streaming; cache; prefill; low-RAM mode:
- MTP; reasoning; sampling; calibration; experimental speed projection:

```text
Exact launch command and relevant configuration, with credentials removed.
```

## Method

Link prompts and scripts. Describe output cap, repetitions, warm-up, loading,
prompt reuse, expert-cache state, timing boundaries, and memory measurement.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Fill in measured values | | | | | | | |

Link per-run data and logs. Include total latency and memory observations when
measured, with units and timing boundaries. List failures and skipped cases.

## Correctness and limitations

Describe checks, expected answers, observed results, and untested behavior.
````

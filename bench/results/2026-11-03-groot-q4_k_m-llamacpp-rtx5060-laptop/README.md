# Groot Q4_K_M: Strata vs llama.cpp on an RTX 5060 Laptop 8 GB (2026-11-03)

Head-to-head on the same GGUF, same prompt, same sampler settings. Measurement only; script
`bench/bench_vs_llama.py`. One machine, one round per config, 2 reps per temperature; medians over the 2 reps.

## What was compared

- **Model:** `GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf`, 12,484,339,584 bytes, SHA-256
  `c941e12a231b40b8fb9f6ad7f6601878d72db39c2c62aefb185fbe5c7d0c2e29`; both engines read the same file. The embedded
  MTP block is used by both.
- **Strata:** ready-made `strata-poly` 0.1.45 (CUDA 13.0), branch `add-groot-family`. Config as setup writes it:
  32,768-token context, MTP on, conversation cache `auto`, 15 threads. The `--no-draft` arm adds only that flag.
  (`--no-draft` turns off the MTP block; the engine's n-gram lookup still runs, which is why its `draft` column is
  not empty. Treat it as "no MTP", not as plain decoding.)
- **llama.cpp:** built from source here at the pinned commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`
  (version 0.4.1-dev), `-DGGML_CUDA=ON`, arch 120, CUDA 13.1, MSVC 19.44. The vendored source copy has no
  `tools/ui`, so the temporary build removed the UI target and used a stub header with no assets; that changes no
  text path. `-DLLAMA_CURL=OFF`, tests off. Default build flags otherwise.
- **llama.cpp settings:** `--fit on -c 32768 -fa on` (the `fit` arm), the same with `--fit-target 512` (the
  `fit512` arm, which leaves less VRAM free and was faster), and `--spec-type draft-mtp --spec-draft-n-max 3` for
  the MTP arm.
- **Protocol:** `/v1/chat/completions` on both servers, thinking off, 256 new tokens, `top_k 20, top_p 0.95` at
  temp 0.7, seed 1, prompt cache off, a unique nonce per request. The prompt is the first 4,000 Groot tokens of this
  repository's docs plus "Summarize the text above in three sentences."
- **Interleaving:** the engines cannot be loaded together, so a config is started, warmed with two short requests,
  run through all cells, and stopped. VRAM is the whole-machine `nvidia-smi` peak; RSS is the engine process tree.
  The laptop drives its display from the same card (~0.5 GiB).

## Results (4K prompt, medians of 2)

| config | temp | prompt tok/s | decode tok/s | TTFT s | MTP accepted | VRAM peak MiB | RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Strata, MTP** | 0.0 | **1,881** | **80.7** | **2.15** | 0.70 | 7,094 | 10.0 |
| **Strata, MTP** | 0.7 | **1,860** | **86.0** | **2.18** | 0.76 | 7,094 | 10.0 |
| Strata, `--no-draft` | 0.0 | 1,889 | 68.4 | 2.15 | 0.43 | 7,062 | 8.2 |
| Strata, `--no-draft` | 0.7 | 1,834 | 70.3 | 2.21 | 0.39 | 7,062 | 8.2 |
| llama.cpp `fit` | 0.0 | 597 | 47.5 | 6.78 | - | 6,152 | 12.4 |
| llama.cpp `fit` | 0.7 | 582 | 48.0 | 6.95 | - | 6,152 | 12.9 |
| llama.cpp `fit512` | 0.0 | 625 | 50.0 | 6.49 | - | 6,658 | 12.3 |
| llama.cpp `fit512` | 0.7 | 618 | 51.0 | 6.56 | - | 6,658 | 12.9 |
| llama.cpp `fit` + MTP n=3 | 0.0 | 519 | 52.3 | 7.81 | 0.47 | 6,132 | 12.8 |
| llama.cpp `fit` + MTP n=3 | 0.7 | 523 | 49.0 | 7.74 | 0.41 | 6,132 | 13.3 |

Strata leads every row. Against the best llama.cpp arm (`fit512`): prompt **3.0×** (1,881 vs 625), decode
**1.6–1.7×** (80.7 vs 50.0 and 86.0 vs 51.0), and time to first token **3.0×** (2.15 s vs 6.49 s). Against
llama.cpp's MTP arm the decode gap is 1.5–1.8×.

- **MTP helps both, by different amounts.** Strata's decode rises 68.4 → 80.7 tok/s (+18%) with it; llama.cpp's
  rises 47.5 → 52.3 (+10%) at temp 0 but falls to 49.0 from 51.0 at temp 0.7. Strata's draft accepts 0.70–0.76 of
  its guesses, llama.cpp's 0.41–0.47.
- **Prompt speed is where the gap is widest.** The same 4,000-token prompt reads in 2.1 s on Strata and 6.5 s on
  llama.cpp. llama.cpp's prefill is not tuned here beyond `--fit` (no `-b`/`-ub` sweep, no per-tensor overrides),
  so some of this gap is settings, not architecture.
- **Memory.** llama.cpp put 0.4–0.9 GiB less of the model in VRAM (6.1–6.7 GiB peak vs Strata's 7.1 GiB) and still
  decoded slower. Its RSS is higher because it mmaps the file; Strata's figure is the process tree's resident set.

## Caveats (be fair)

- One round, 2 reps per temperature. The medians are over 2 numbers, not 10; the raw rows are in `raw/`.
- Only the 4K cell was run. A 32K cell does not fit the time budget on this laptop today (llama.cpp reads 32K at
  roughly 500 tok/s, ~65 s a request, and Strata's expert cache behaviour at 32K is the subject of the 2026-11-02
  benchmark).
- llama.cpp tuning was `--fit` and `--fit-target` only: no `-b`/`-ub` sweep, no `-ot` per-tensor overrides, no
  `--no-mmap`/`--load-mode none` trial. In the 2026-10-06 comparison, a tuned `-ncmoe` beat `--fit` by about 2× on
  prompt speed on a 12 GB RTX 3060 and a 16 GB P100; a tune here could narrow the prompt gap, though those cards had
  more VRAM per expert. No decode setting is obviously missing.
- The llama.cpp MTP arm is the model's own embedded block (llama.cpp reports draft acceptance, so it loaded), not a
  separate head file.
- Both servers ran with their default KV cache type (f16) and `-fa on`; KV quantization was not compared.
- `fit512` and the other arms were measured in separate sessions on the same day, after the first round; the machine
  was otherwise idle apart from the display.

## Quality sanity

`quality` mode, greedy, 200 tokens, same two prompts through all five configs (`raw/quality.json`):

- The code answer ("a Python function that parses a CSV file into a list of dicts, with tests") is **identical
  character for character** between Strata with MTP and llama.cpp `fit512` — all 813 characters. Both open the same
  way against the other arms too.
- The story ("a short story about a lighthouse keeper") is identical for the first 165 characters, then diverges:
  Strata continues "For the keepers of the world, this was not a moment of stillness", llama.cpp `fit512` "For the
  keepers of the world, the light was not a luxury". Both stay coherent and on-task; this is the expected drift from
  different floating-point orders in the quantized matmuls, not a broken answer.
- No arm produced repetition, garbage or a cut-off answer. The `--no-draft` and MTP arms of Strata differ from each
  other in the same way.

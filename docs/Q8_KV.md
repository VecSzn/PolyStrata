# The Q8_0 KV cache (`strata-poly --kv q8_0`)

`strata-poly` can store its attention K and V caches in the ggml **Q8_0** format instead of fp16. It is
opt-in: `--kv q8_0`. The default, `--kv fp16`, is unchanged and byte-identical to the build before this
option existed. The format halves the KV cache's VRAM and, on the models tested, leaves the answers alone.

Gemma 4 has its own two, in the same Q8_0 rule but its own layout: `--kv int8` and `--kv int8k`
([the int8 pair, below](#the-int8-pair-for-gemma-4-strata-poly---kv-int8-and---kv-int8k)).

On `kolibri1`, `STRATA_KV_INT8=1` makes both the decode and the prompt read the q8_0 cache with the int8
reader below (`attend_int8`) instead of the float q8 readers. It is experimental: the queries are quantized
too, so the answers change slightly, and it is opt-in.

## What the format is

A cache row holds one value a token for each KV head and each head dimension (`n_kv * head_dim` values).
Stored as fp16 that is `2 * n_kv * head_dim` bytes a token for K and, again, for V. Stored as Q8_0 it is cut
into 32-value blocks: for every block of 32 values, one fp16 **scale** (the block's largest magnitude over 127)
and 32 **int8 codes**. That is `n_kv * (head_dim / 32) * 2` bytes of scales plus `n_kv * head_dim` bytes of codes,
a row for K and the same for V. Attention dequantizes a block with its scale on the fly while it computes the
scores and the outputs.

For the three tested models (`n_kv = 2`, `head_dim = 256`, 11 attention layers for Tiel/Ornith, 10 for Occamy)
the engine prints its row size at load:

```
strata-poly: q8_0 KV cache: 544 bytes a row
```

544 = 512 codes + 2 blocks × 2 scales, a head, K and V. `--kv q8_0` needs `n_kv * head_dim` to be a multiple
of 32; the models here are. A grammar that is not gets a clear error rather than a wrong answer.

All three models loaded and answered with `--kv q8_0` at the full 131072-token context, and with a picture
through the vision encoder (`strata-vision`), each replying `KV OK` to the fixed prompt.

## What it was measured on

| | |
| --- | --- |
| Graphics card | NVIDIA RTX 3070, 8192 MiB |
| CPU | 24 threads |
| Built with | CUDA (the CUDA build of `strata-poly`); `/opt/cuda` |
| Models | `Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf`, `Ornith-1.5-35B-A3B.APEX-I-MiniPlus-V2.1.gguf`, `Occamy-1.0.APEX-I-MiniPlus-V2.1.gguf`, all `qwen35moe`, Q4_K - Medium |
| Context | 131072 tokens |
| Server | `serve/server.py --engine strata --config strata-<model>.json --port 8099` |

## The answers do not move

With greedy sampling (`temperature: 0`, which the engine reads as its own greedy), `--no-draft` so the
speculative path cannot change a token (`timings.draft_n` was 0 on every request), and `max_tokens: 64`, the
fp16 and Q8_0 arms were compared over five prompts each: the four short factual ones and one open-ended
sentence about the colour blue.

| model | exact matches | mean character agreement | longest common prefix |
| --- | ---: | ---: | --- |
| Tiel Coder 35B-A3B | 5/5 | 1.0000 | full reply (137 chars) |
| Ornith 1.5 35B-A3B | 5/5 | 1.0000 | full reply (129 chars) |
| Occamy 1.0 | 5/5 | 1.0000 | full reply (173 chars) |
| **all three** | **15/15** | **1.0000** | — |

Every Q8_0 answer equals the fp16 one character for character. The short factual prompts (`KV OK`, `42`,
`Paris`, `2, 3, 5, 7, 11`) also equal the pre-change fp16 baseline byte for byte on all three models
(12/12), which is how the default path's byte-identity was checked; the open-ended fifth prompt is not
bit-stable at greedy on Ornith and Occamy even before this change (speculative drafts on Ornith, `--tail-skip`
and the adaptive expert cache on Occamy), so it is not used for that check. The engine/API does not return
tokens, so agreement is measured on the reply text (longest common prefix and per-prompt agreement).

## What it saves

Per token, and in total at a full 131072-token context:

| model | attention layers | Q8_0 a token | fp16 a token | Q8_0 at 131072 | fp16 at 131072 | saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiel | 11 | 11 968 B | 22 528 B | 1.461 GiB | 2.750 GiB | 1.289 GiB |
| Ornith | 11 | 11 968 B | 22 528 B | 1.461 GiB | 2.750 GiB | 1.289 GiB |
| Occamy | 10 | 10 880 B | 20 480 B | 1.328 GiB | 2.500 GiB | 1.172 GiB |

**46.88% less KV VRAM** for all three, matching the per-token arithmetic exactly.

The engine's default KV cache grows as the conversation does, so a short prompt barely moves the load-time
expert cache — but with `--kv-whole`, which reserves the whole context's KV at load, the freed VRAM goes to
experts. At 131072 the expert cache under `--kv-whole` rose

```
Tiel   fp16: 1237 of 10496 experts (1.5 GiB)   q8_0: 2319 of 10496 (2.8 GiB)   +1082 experts
Ornith fp16: 1237 of 10496 experts (1.5 GiB)   q8_0: 2313 of 10496 (2.8 GiB)   +1076 experts
Occamy fp16: 1503 of 10240 experts (1.8 GiB)   q8_0: 2480 of 10240 (2.9 GiB)    +977 experts
```

with the same free VRAM left per model at the end.

## What it costs

At a 972-token prompt, 128-token answers, greedy, `--no-draft`, `--vram-reserve-mib 512` (see the caveat),
each arm a fresh process, median of four steps:

| model | arm | prompt tok/s | decode tok/s |
| --- | --- | ---: | ---: |
| Tiel | fp16 | 1014.6 | 105.8 |
| Tiel | q8_0 | 1008.0 | 105.2 |
| Ornith | fp16 | 1022.8 | 106.5 |
| Ornith | q8_0 | 1008.4 | 106.2 |
| Occamy | fp16 | 1031.8 | 107.1 |
| Occamy | q8_0 | 1034.1 | 106.9 |

Q8_0 against fp16: prompt **−1.4% to +0.2%**, decode **−0.6% to −0.2%** — within run-to-run scatter.

## A caveat that is not about Q8_0

At 131072 with the default 256 MiB reserve, long chats intermittently fail with
`cudaGraphInstantiate: out of memory`: the engine grows the expert cache down to 128 MiB free, then a newly
recorded CUDA graph needs more than what is left. This is **pre-existing and general**: it reproduces on the
unchanged fp16 path too, so it is not caused by `--kv q8_0`. `--vram-reserve-mib 512` removes it — all four
model/format combinations, eight requests each, with and without a picture, saw zero OOM. The numbers above
were taken with that 512 MiB reserve.

The three run configs (`strata-tiel.json`, `strata-ornith.json` and `strata-occamy.json`) now ship with
`--vram-reserve-mib 512`, so the OOM does not occur with them. With that reserve, Tiel at the full
131072-token context loads 3481 of 10496 routed experts into VRAM (4.1 GiB), measured on the RTX 3070 below.

## What was left alone

- The `strata` engine. Nothing in it changed.
- `glm5next` / GLM 5.3 Flash (`dsa_attend`): its attention is not the one this cache quantizes, so neither
  `--kv q8_0` nor `--kv int8` is offered for it.
- `--kv q8_0` and a `gemma4` file: still rejected cleanly at load
  (`--kv q8_0 is only supported for qwen35moe models in v1; this is a 'gemma4' model`), not run.
- ROCm/HIP: the same kernels, measured below on an AMD card (it was rejected on HIP until that was done).
- The default path stays byte-identical to the last release.

## The comparison with another implementation

`llama.cpp` (the in-tree build `runners/llama.cpp/build/bin/llama-cli`) reads the same `qwen35moe` files and
accepts `-ctk q8_0 -ctv q8_0`. Given a Q8_0 KV cache it answered the two short fixed prompts
(`Reply with exactly: KV OK` → `KV OK`; `What is 17 + 25? Answer with just the number.` → `42`), the same
tokens the poly engine's `--kv q8_0` produced on all three models. It was run on the CPU (`-ngl 0`) because the
file is far larger than the card; only the short deterministic answers are a meaningful signal across engines.
Full transcript in the T8.5 report.

## Files touched

- `src/poly/engine.hpp` — `KvFmt` and `Options::kv_format` (default `kFp16`).
- `src/poly/main.cpp` — `--kv fp16|q8_0` parsing and help text.
- `src/poly/engine.cpp` — the `qwen35moe`-only check and the clean rejection for other architectures.
- `src/poly/qwen35.hpp`, `src/poly/qwen35.cpp` — the Q8_0 cache pointers, allocation, save/load, the row-size
  print, and the attention path that reads codes+scales.
- `src/poly/kernels.cu`, `include/strata/poly/kernels.hpp` — `store_q8` / `store_q8_at` and the Q8_0 attention
  readers (scores and outputs).
- `src/poly/kv_q8_test.cpp` (new) — the kernel unit test: round-trip within half a block scale, byte-exact
  layout, and the ring store.
- `CMakeLists.txt` — the `poly_kv_q8_test` target and its `add_test`.
- `strata-tiel.json`, `strata-ornith.json`, `strata-occamy.json` — the three run configs now carry
  `--kv q8_0` and `--vram-reserve-mib 512`.
- `gemma4.hpp` / `gemma4.cpp` — untouched.

## Builds

- **CUDA:** builds clean; the test-suite run is **116/117**. The one failure, `ple_parity`, is a missing model
  file (`.../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf: cannot open`), unrelated to this change.
  `poly_kv_q8_test` passes (0.23 s), `poly_kv_int8_test` 35/35 (0.4 s).
- **HIP:** builds clean for gfx1201; `poly_kv_q8_test` passes (91,530 values checked) on an RX 9070 XT. The
  int8 family builds there too: the MMA tier's PTX has an empty body and `attend_int8` never selects it
  (`tensor_cores()` is false under `__HIPCC__`), so the dp4a tier and the fp16 twin carry the prompt path;
  `poly_kv_int8_test` skips the mma probes with `STRATA_USE_HIP` set.

## On an AMD card

RX 9070 XT (16 GB, gfx1201), Ryzen 7 9800X3D, 30 GB of RAM, Fedora, ROCm 7 from AMD's wheels;
`Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`, `--kv-whole`, a 2 GiB conversation cache. fp16 at 131072 against Q8_0
at 196608, each a fresh server:

| | experts in VRAM | 256-token answer | 4.5K prompt | 64K prompt | answer after it |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16, 131072 | 4415 (7.9 GiB) | 110-127 tok/s | 1,814-1,822 tok/s | 2,065 tok/s | 66 tok/s |
| Q8_0, 196608 | 4729 (8.4 GiB) | 106-130 tok/s | 1,843-1,862 tok/s | 2,075 tok/s | 63 tok/s |

Greedy with `--no-draft`, seven prompts: the four short ones gave the same answers; a docstring, a two-sentence
summary of a 24K-character prompt and a list of names from a 60K-character one took other words from the 249th,
14th and first character on, each answer as right as the fp16 one.
- **SYCL:** not applicable; that is a separate engine.

## The int8 pair for Gemma 4 (`strata-poly --kv int8` and `--kv int8k`)

Same rule, different layout. The block's largest magnitude over 127 is the **delta**, `roundf(x / delta)` is the
**code** — llama.cpp's Q8_0 rule, which is what the engine already uses for its expert weights
(`ggml_quantize_chunk`, `GGML_TYPE_Q8_0`). What changes is where the bytes go: instead of ggml's 34-byte
`block_q8_0` (a delta then 32 codes), a row is `n_kv * head_dim` **codes back to back** and, beside them,
`n_kv * head_dim / 32` **halves** — one delta a 32-value block.

Two arrays because a lane's four codes are then one aligned load (`char4`, 4-aligned). In ggml's layout half
the blocks put the codes at an address 2 mod 4, so a `__dp4a` has to `memcpy` four bytes into an int first.

The two formats:

| | what is quantized | what `store_int8` writes |
| --- | --- | --- |
| `--kv int8` | both sides | codes and deltas, K and V |
| `--kv int8k` | the keys' side only | codes and deltas for K; V stays fp16 |

A decode step is a `__dp4a` of a lane's own run of codes — the query's against the key's, both 4-aligned — and
the block's two deltas scale the integer sum once: llama.cpp's `vec_dot_q8_0_q8_1`, whose point is that
neither side is ever turned into floats. Above 16K keys the scores run on an int8 tensor-core MMA
(`mma.sync.m16n8k32.s8`, the readers' shapes need no shuffle); below, and on any card without it, dp4a.
A prompt batch is the other way round: the codes are converted to halves once per batch — O(rows) against an
attention's O(rows²) — and cuBLAS takes the products from there, as llama.cpp does on CUDA
("other kernels convert these to f16"). `attend_int8_mixed` reads the K side's codes for the scores and the
fp16 V cache for the weighted sums.

`--kv int8` needs `n_kv * head_dim` to be a multiple of 32 and each head to be 64 values or 128 and up: the
score readers give a lane a whole number of dp4a steps that lies inside one block. A head of another shape
gets a clear error at load (`unsupported model: --kv int8 needs a head of 64 values or of 128 and up ...`).
Both formats are Gemma 4's: any other architecture is rejected at load
(`--kv int8 and --kv int8k are Gemma 4's; for a qwen35moe model use --kv q8_0`), because the two readers'
arithmetic is not the same and a model takes one family or the other.

### What was measured

Kernels, without a model (`poly_kv_int8_test`, 35 checks, RTX 4070 Laptop): the codes and deltas agree with
ggml's own `quantize_row_q8_0_ref` **to the bit** (0 of 1,536,000 halves differ for the whole-cache
dequantizer); two layers' shapes give `cos(attend, attend_int8)` 0.99995-0.99996 at 64 to 3000 keys, the
window layers' rings included; the MMA probe runs one product and checks it (0 of 128 wrong), and the MMA is
3.4× the dp4a where it runs.

End-to-end on Gemma 4 (26B, `Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf`, Q4_K_M), on an
RTX 4070 Laptop (8 GiB), 13 threads, `--no-draft --window 1 --tail-skip 0 --expert-swaps 0 --cpu-share 0
--vram-reserve-mib 512`, the expert split held still (`STRATA_FREEZE_EXPERTS=1`), greedy, five prompts
(the four short factual ones and one open-ended sentence about the colour blue), each arm a fresh process:

| | prompt, ms (median of 15) | decode, tok/s (median of 15) |
| --- | ---: | ---: |
| fp16 | 984.0 | 37.68 |
| int8 | 941.8 (−4.3%) | 38.22 (+1.4%) |
| int8k | 968.1 (−1.6%) | 38.01 (+0.9%) |

The three arms are within run-to-run scatter of each other, the int8 ones a little ahead: the file is
Q4_K_M, so most of the model's work is elsewhere, and this card cannot hold the experts either way
(900 of 3840 routed experts in VRAM; the rest on the CPU). The KV cache is a small part of what this
configuration reads, which is why the saving does not show up as a speed-up here. The point of the format is
the VRAM it gives back, not the clock.

The answers: the four factual prompts keep their answer under both formats (`KV OK`, `42`, `Paris`,
`2, 3, 5, 7, 11`), the difference confined to the chain-of-thought phrasing; the open-ended one is a
creative prompt, and each format words it differently.

The same code on another machine (the `pr3` branch, before this change renamed the symbols) writes the
**same 64 tokens** for all five prompts under both `int8` and `int8k` (10 of 10 runs identical, and
2704 of 2704 `--dump-layers` lines identical), so the rename and the split of the two families changed
no arithmetic.

### Files touched

- `src/poly/kernels.cu`, `include/strata/poly/kernels.hpp` — `KvInt8`, `kQ8_0Values`, `store_int8` /
  `store_int8_at` / `dequant_int8_to_f16`, and `attend_int8` / `attend_int8_mixed` with their score readers
  (dp4a, register, and MMA tiers). Nothing of the Q8_0 family beside them.
- `src/poly/gemma4.hpp` / `gemma4.cpp` — the codes/delta arrays beside the fp16 ones, the fp16 twin for a
  prompt batch and its per-sequence rows, the ring stores, and the save/load and parking paths
  (`card_row_bytes`, `save_card_rows` / `load_card_rows`, `extra_bytes`, `save_layer` / `load_extra`).
- `src/poly/engine.hpp` — `KvFmt` grows `kInt8` and `kInt8K`, with the `int8()` / `int8k()` readers.
- `src/poly/engine.cpp` — the clean rejection of int8/int8k for anything but `gemma4`.
- `src/poly/main.cpp` — `--kv int8` / `--kv int8k` parsing and the help text.
- `src/poly/kv_int8_test.cpp` (new) — the kernel unit test.
- `CMakeLists.txt` — the `poly_kv_int8_test` target and its `add_test`.

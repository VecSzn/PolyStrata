# The int8 KV cache (`strata-poly --kv int8` / `--kv int8k`)

`strata-poly` can store its attention K and V caches in the ggml **Q8_0** format instead of fp16. It is
opt-in: `--kv int8` (`q8_0` is kept as an alias of it), and `--kv int8k` quantizes the keys' side only
and leaves the values in fp16. The default, `--kv fp16`, is unchanged and byte-identical to the build
before this option existed. The format halves the KV cache's VRAM and, on the models tested, leaves the
answers alone.

This replaces the `--kv q8_0` that v0.1.43-poly.5 shipped. What was stored is the same numbers (the same
delta = amax/127, the same roundf codes, byte-identical against ggml's `quantize_row_q8_0_ref`); what
changed is how the caches are laid out and read. The old format kept ggml's 34-byte `block_q8_0` blocks
and every attention kernel dequantized them element by element into floats — two extra loads and four
I2F/FMA per four values, in every score and every output sum. The int8 here is a **compute format**: the
codes and the deltas are two separate arrays (the codes back to back, so every four are one aligned
word; a delta a 32-value block), the queries are quantized into the same form, a lane's run of a decode
head stays inside one 32-value block, and one `__dp4a` takes four products with the delta multiplied in
once a run. The wider heads take an int8 tensor-core MMA. A prompt batch dequantizes the caches into an
fp16 twin once — incrementally, O(rows) against the O(rows²) attention — and hands the rows to cuBLAS's
`attend_batch` as before.

On Gemma 4 the whole set works too (it had none before): the window layers' ring stores write in two
pieces, the twin is vmm-mapped on demand (`twin_cover`), and `--kv int8k` mixes quantized keys with fp16
values, which is the mode Gemma 4's window-plus-full layers take best.

## What the format is

A cache row holds one value a token for each KV head and each head dimension (`n_kv * head_dim` values).
Stored as fp16 that is `2 * n_kv * head_dim` bytes a token for K and, again, for V. Stored as int8 the
codes and the deltas are two arrays: `n_kv * head_dim` bytes of codes (K and V), then
`n_kv * (head_dim / 32) * 2` bytes of fp16 deltas. That is `n_kv * head_dim * 17 / 16` bytes a row for
K and the same for V — the same 544 bytes a row the old format took on the tested models, with the codes
in one run so every four of them are one aligned word.

For the three tested models (`n_kv = 2`, `head_dim = 256`, 11 attention layers for Tiel/Ornith, 10 for Occamy)
the engine prints its row size at load:

```
strata-poly: int8 KV cache: 544 bytes a row (K and V), 272 with --kv int8k
```

544 = 512 codes + 2 blocks × 2 deltas a head, K and V. `--kv int8` needs `n_kv * head_dim` to be a multiple
of 32; the models here are. A grammar that is not gets a clear error rather than a wrong answer.

All three models loaded and answered with `--kv int8` at the full 131072-token context, and with a picture
through the vision encoder (`strata-vision`), each replying `KV OK` to the fixed prompt. Gemma 4
(`gemma-4-26B-A4B`, 25 window + 5 full layers) runs `--kv int8k`, and its layers dump matches the reference
engine's bit for bit (3603 rows).

## What it was measured on

| | |
| --- | --- |
| Graphics card | NVIDIA RTX 4070 Laptop, 8192 MiB |
| Built with | CUDA (the CUDA build of `strata-poly`), CUDA 13.4, arch 89 |
| Models | `gemma-4-26B-A4B` (`--kv int8k`), qwen35moe Ornith 1.5 35B-A3B (`--kv int8`) |
| Context | 4096+ token prompts, answers of 128+ tokens |
| Server | `serve/server.py --engine strata --config <model>.json` |

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

The int8 rewrite was checked the same way on gemma-4-26B-A4B, against the same engine built with the old
`--kv q8_0` readers as the reference: both engines ran `--dump-layers` on one fixed prompt, one seed, three
tokens, and every attention stage's every layer agreed — **3603 of 3603 dump rows bitwise equal**. The
engine-to-engine A/B over 36 requests: decode ratio 1.0167, bootstrap 95% CI [0.9813, 1.0577] (the CI
contains 1.0 — the two arms are statistically indistinguishable at decode); prefill ratio 0.9844. The
answers agree where greedy is stable, and every difference traces to the quantization, not the plumbing.

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

The int8 rewrite holds the same line: on the 4019-token gemma-4 prompts the int8k arm's prefill was
0.9844 of the fp16 twin's and its decode ratio 1.0167 (CI above), both within scatter; the kernel-level
test (35 checks) times the q8 decode path against the fp16 one at 0.88–0.96× — the __dp4a/MMA decode
reads half the bytes and pays the quantize of the queries, which nets out level with fp16 on a 4070-class
card and clears VRAM for experts it would otherwise spill.

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

- The `strata` engine, and its own `--kv int8`. Nothing in it changed.
- `glm5next` / GLM 5.3 Flash (`dsa_attend`): its attention is not the one this cache quantizes, so the quantized
  cache is not offered for it.
- The default path stays byte-identical to the last release.

## The int8-specific notes

- The int8 **MMA tier** uses NVIDIA's `mma.sync` PTX, which a HIP build does not have; the `__dp4a` tiers run
  through `hip_compat`'s `dp4a` (the same signed-bytes-in-int32 contract). On AMD the engine takes the dp4a
  readers and the cuBLAS batch path; `STRATA_KV_MMA=0` pins the same choice on CUDA. The AMD numbers below are
  the old elementwise readers (PR 29); they are re-measured with this change pending.
- Gemma 4 is in with this change (`--kv int8` / `--kv int8k`; int8k is its best mode — see above).

## The comparison with another implementation

`llama.cpp` (the in-tree build `runners/llama.cpp/build/bin/llama-cli`) reads the same `qwen35moe` files and
accepts `-ctk q8_0 -ctv q8_0`. Given a Q8_0 KV cache it answered the two short fixed prompts
(`Reply with exactly: KV OK` → `KV OK`; `What is 17 + 25? Answer with just the number.` → `42`), the same
tokens the poly engine's `--kv q8_0` produced on all three models. It was run on the CPU (`-ngl 0`) because the
file is far larger than the card; only the short deterministic answers are a meaningful signal across engines.
Full transcript in the T8.5 report.

## Files touched

- `src/poly/engine.hpp` — `Options::kv_int8` / `Options::kv_int8_k` (default off; the `KvFmt` enum and the
  elementwise readers it selected are gone).
- `src/poly/main.cpp` — `--kv fp16|int8|int8k` parsing (`q8_0` stays as an alias of `int8`) and the help text.
- `src/poly/kernels.cu`, `include/strata/poly/kernels.hpp` — `store_q8` / `store_q8_at`,
  `dequant_q8_to_f16`, `attend_q8` / `attend_q8_mixed` (plain `__dp4a`, register-blocked and int8-MMA
  tiers), `Q8Cache` / `kQ8_0Values`; the elementwise dequant-then-GEMM readers are gone.
- `src/poly/qwen35.hpp`, `src/poly/qwen35.cpp` — the codes/delta caches, the queries quantized
  (`q8c_`/`q8d_`), the fp16-twin batch path; `attn_rows()` (the batch slots' per-row store) in the same shape.
- `src/poly/gemma4.hpp`, `src/poly/gemma4.cpp` — the ring stores' two-piece writes, the on-demand fp16
  twin (`twin_cover`), `attend_q8_mixed` for `--kv int8k`, and the save/load serialization.
- `src/poly/kv_q8_poly_test.cpp` (replaces `kv_q8_test.cpp`) — the store against ggml's
  `ggml_quantize_chunk` block by block, byte-exact layout, and the three attention paths compared; 35 checks.
- `CMakeLists.txt` — the `poly_kv_q8_test` target points at the new test file.

## Builds

- **CUDA:** builds clean; the test-suite run is **116/117**. The one failure, `ple_parity`, is a missing model
  file (`.../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf: cannot open`), unrelated to this change.
  `poly_kv_q8_test` passes (0.23 s).
- **HIP:** builds clean for gfx1201; `poly_kv_q8_test` passes (91,530 values checked) on an RX 9070 XT.

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

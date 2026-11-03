# LFM2.5-24B-A2B (`strata-poly`)

`strata-poly`, the engine of GLM 5.3 Flash and the smaller models, also runs Liquid AI's LFM2.5-24B-A2B: llama.cpp's
architecture `lfm2moe`, read from the GGUF files of Unsloth and Liquid AI. The server, the web app and the API are the
ones the README describes.

It is new here. This project has run it on two machines: a Windows laptop whose card has 8 GB, and a Linux
workstation with a 32 GB card.

Its license is Liquid AI's LFM Open License v1.0, not an open-source one: commercial use is licensed only to
companies under 10 million US dollars of yearly revenue
([the license](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)).

## The sizes

| Size | From | Download | Experts | RAM beside an 8 GB card (setup's estimate) |
| --- | --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | [Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF): 4-bit experts, 5- and 6-bit where they write | 14.9 GB | 14.0 GB | about 20 GB |
| **UD-Q3_K_XL** | Unsloth: 3-bit i-quant experts, 4-bit where they write | 10.9 GB | 10.2 GB | about 16 GB |
| **QAD-Q4_0** | [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF): Q4_0, trained for it (quantization-aware distillation) | 13.6 GB | 12.9 GB | about 18 GB |
| **UD-Q6_K_XL** | Unsloth: 6-bit experts, 8-bit where they write | 21.9 GB | 20.9 GB | about 27 GB |
| **Q8_0** | Unsloth: 8-bit throughout | 25.5 GB | 24.4 GB | about 31 GB |

Setup takes UD-Q4_K_XL, or UD-Q3_K_XL when the PC has too little RAM for it; the others are chosen by name
(`--model QAD-Q4_0`). Every file is pinned to its repository's commit of 2026-10-15 and checked against its SHA-256.

## How fast

The README's numbers are PolyStrata's own benchmark through the server, the test every model of this engine gets
([BACKENDS.md](BACKENDS.md#a-prompt-of-30000-tokens)): `polystrata.py benchmark --prompt-tokens 24000`, a short chat
and a prompt built from this repository's documents, 256 tokens written after each, three runs a size and the last
two below. The engine is 4a644a8d's, with the run configs setup writes for each machine: on both, a prompt is read
2,048 tokens at a time.

### On a laptop with 8 GB of VRAM

An RTX 4070 Laptop with 8 GB, an i7-14700HX, 32 GB of RAM, Windows 10, CUDA 13.0. A context of 32,768 and 13 threads
for the experts the card does not hold. The prompt came to 29,804 tokens.

| Size | First token, s | Reads, tokens/s | Writes a short chat | Writes after the prompt | Experts from VRAM | Machine's RAM, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | 8.3 to 8.4 | 3,575 to 3,601 | 67 to 68 | 69 | 52% to 53% | 22.5 to 22.6 |
| UD-Q3_K_XL | 7.9 to 8.0 | 3,758 to 3,811 | 80 to 81 | 70 to 76 | 67% | 18.7 to 18.8 |
| QAD-Q4_0 | 8.1 | 3,685 to 3,694 | 72 to 73 | 77 to 79 | 59% | 21.1 to 21.2 |
| UD-Q6_K_XL | 9.6 to 9.9 | 3,035 to 3,115 | 41 to 44 | 53 to 54 | 36% | 29.1 |
| Q8_0 | 10.3 to 12.4 | 2,406 to 2,923 | 19 to 34 | 44 to 47 | 30% to 31% | 31.2 |

Q8_0 fills the machine's RAM, and part of its file is read from the disk again; that is what makes it the slowest
here, and the most uneven. Four passes before this one, with earlier engines, read the prompt at 1,290 to 3,778
tokens/s, the last of them, d74c5e25's with the same batch, at 2,954 to 3,778. A README cell has the highest of the
five passes: the short chat came a little faster in an earlier one with most sizes.

### On a workstation

An RTX 5090 (32 GB), 25 vCPU of a Xeon Platinum 8470Q, 120 GB of RAM, Linux, CUDA 12.8. The card holds every expert
of every size. The prompt came to 29,816 tokens.

| Size | First token, s | Reads, tokens/s | Writes a short chat | Writes after the prompt | Experts from VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | 1.3 to 1.4 | 23,387 to 23,462 | 480 to 485 | 639 to 640 | 100% |
| UD-Q3_K_XL | 1.3 to 1.4 | 22,923 to 23,396 | 455 to 472 | 663 to 675 | 100% |
| QAD-Q4_0 | 1.3 | 23,878 to 23,899 | 495 to 497 | 419 to 420 | 100% |
| UD-Q6_K_XL | 1.5 | 21,252 to 21,270 | 423 to 426 | 366 to 376 | 100% |
| Q8_0 | 1.4 | 23,272 to 23,349 | 376 | 341 to 343 | 100% |

The answer after the prompt goes faster than the short chat where it repeats the prompt's text and the engine's
guesses from it are right. One token at a time (`--window 1`), UD-Q4_K_XL writes at 452 tokens/s after a short
prompt and at 387 after 29,816 tokens. Three passes before this one, with earlier engines, read the prompt at 10,007
to 21,666 tokens/s; a README cell has the highest of the four.

### Against llama.cpp

On the laptop, UD-Q4_K_XL through each server with the same prompts
([the runs](../bench/results/2026-10-15-laptop-lfm25-llamacpp-b11540/README.md)):

| | PolyStrata | llama.cpp b11540, `--n-cpu-moe 25` | llama.cpp b11540, `--fit on` |
| --- | ---: | ---: | ---: |
| Reads about 4,000 tokens | 2,193 to 2,215 tokens/s | 498 to 505 | 483 to 485 |
| Writes after them | 89 to 94 tokens/s | 57.7 to 58.0 | 55.1 to 55.3 |
| Reads about 32,000 tokens | 3,543 to 3,545 tokens/s | 528 to 537 | 505 to 516 |
| Writes after them | 71 to 73 tokens/s | 52.8 to 53.0 | 50.9 to 51.0 |

`--n-cpu-moe 25` was the fastest of the settings tried for llama.cpp there. On the workstation, `llama-bench` of the
llama.cpp that [Strata](https://github.com/Niko1221/Strata) v0.1.39 vendors, every layer on the card and flash
attention on:

| | PolyStrata | llama.cpp, batch of 512 (its default) | llama.cpp, batch of 2,048 |
| --- | ---: | ---: | ---: |
| Reads 29,816 tokens | 23,400 tokens/s (18,700 with a batch of 1,024) | 13,311 | 23,614 |
| Writes, one token at a time | 452 tokens/s | 455 | |
| Writes after 29,816 tokens, one at a time | 387 tokens/s | 369 | |

With a batch of 2,048 llama.cpp reads the prompt as fast as PolyStrata, 1% faster, and writes as fast one token
at a time; after the long prompt PolyStrata writes 5% faster. Its guesses add to that where the answer repeats the
prompt.

## Long contexts

LFM2.5 was trained for 131,072 tokens, and setup offers a context up to that. The test is the one DEEPSEEK.md has:
a document of this repository's docs with three facts planted at 15%, 50% and 85% of it, then three questions.
UD-Q4_K_XL, a context of 131,072 and prompts read 2,048 tokens at a time:

- On the workstation, through the server, the 123,609-token request was read and answered in 8.7 s. The answer had
  the second and the third fact and made the first one up.
- Given the prompt as llama.cpp tokenizes it, 123,602 tokens with thinking off, the engine wrote all three facts
  greedily. llama.cpp's own CUDA build (llama-server of 86ebfef, flash attention on, the same file) wrote the second
  and the third and made the first one up. The fact furthest back is where the model is least sure.
- On the laptop the same tokens were read in 44.7 and 45.4 s, 2,700 tokens/s, and the answer written at 35 tokens/s;
  its first 80 tokens had the first two facts.

## How it was checked

- The tokenizer gives the same ids as llama.cpp's `llama-tokenize` on 174,442 tokens: three READMEs (English,
  Chinese, Japanese), BACKENDS.md, a Python file and a CUDA file.
- With greedy sampling the engine and llama.cpp on the CPU write the same tokens until two candidates are close: 1,
  39 and 10 of 64 tokens on three short prompts, 20 on a prompt of 17,575 tokens. llama.cpp's CUDA build parts
  from its CPU build at 5, 40 and 23 tokens on the same three. While they agree on the long prompt, the
  probabilities are 0.087 apart at most.
- A window of guesses writes the same tokens as one token at a time, with the experts on the CPU: 300 tokens after
  a prompt of 26 tokens and after one of 4,259.
- `poly_attend_test`, `poly_norm_q8_test`, `poly_shortconv_test` and `poly_route_test` check the kernels written for
  it against the ones they stand for; `serve/test_lfm2.py` the tool calls and the chat template.

## How the sizes compare

Q8_0 wrote 256 greedy tokens on each of five prompts (code, a summary of 4,260 tokens of BACKENDS.md, arithmetic,
a story, a question in Chinese); each other size then followed those tokens, and the share of steps where its most
likely token is Q8_0's was counted. Q8_0 stands in for the model as trained here.

| Size | Same most likely token as Q8_0 | Mean probability of Q8_0's token |
| --- | ---: | ---: |
| UD-Q6_K_XL | 96.8% (1,239 of 1,280) | 0.782 |
| UD-Q4_K_XL | 95.5% (1,222 of 1,280) | 0.781 |
| QAD-Q4_0 | 92.7% (1,186 of 1,280) | 0.762 |
| UD-Q3_K_XL | 91.6% (1,173 of 1,280) | 0.766 |

## What was tuned for it

The decode numbers are UD-Q4_K_XL on the RTX 5090 with the engine's guesses, after a short prompt and after one of
4,300 tokens, tokens/s:

| Commit | Change | Effect |
| --- | --- | --- |
| d4d5049c | the model runs | 386 and 347 |
| 6af34f82 | a layer's residual sum, RMS norm and Q8_1 quantization in one launch | 401 and 359 |
| 962da9b9 | the router's kernel with as few slots a lane as 64 experts need | 405 and 368 |
| 72a738f6, 165bb4da | with every expert on the card, the card picks a few tokens' experts itself, and a token's 40 layers run as one recorded launch | 454 and 408 |
| 4f5947f5 | the attention of a few tokens in blocks of 64 rows, a launch block each | one token at a time after 4,300 tokens 381 to 441, after 30,000 196 to 350; the laptop after 30,000 49 to 56 |
| d74c5e25 | a prompt batch's attention in registers, the way FlashAttention-2 goes | a prompt of 29,816 tokens read in 1,852 ms instead of 2,787 on the 5090, in 8.9 s instead of 16.8 on the laptop |
| 0ec4802d | setup's config reads 2,048 tokens at a time on a card that holds every expert | 1,415 ms instead of 1,852 |
| 36fdc62e | and on any card from 8 GB | the laptop 7.8 to 8.0 s instead of 8.9 to 9.1 |
| 3d51cba9 | the blocks of rows added up 16 warps at a time | one token at a time after 30,000 tokens 348 to 388 |
| cd87aecd | the next block of keys copied in while attn_flash64 works on one | 1,361 to 1,371 ms instead of 1,398 to 1,400 |
| 4a644a8d | a prompt batch's cached experts in three widths of tiles instead of nine | 1,280 ms instead of 1,367 to 1,376 |

## What you need

- An NVIDIA card, RTX 20 or newer, with 8 GB of VRAM or more. About 4 GB go to the weights outside the experts, the
  caches and the buffers at a 32K context; the rest holds experts. A 24 GB card holds the three smaller sizes
  whole, by setup's estimate.
- RAM for the experts the card does not hold. They are read from the mapped file while the model answers, so they
  should stay in the OS file cache: setup's estimate is in the table above. All five sizes have run on the laptop
  with 32 GB, but Q8_0 fills it and is slow there.
- Linux or Windows 10 / 11. AMD cards: the engine compiles for them on Linux and under Windows as
  well (AMD's TheRock wheels); this model has not been tried on one, and setup says so.

## Install

```
./setup.sh --family lfm25
```

On Windows, `START-HERE.bat` and pick it from the list. Setup asks for the size, the context and whether to keep
several conversations in RAM, downloads the file, checks its SHA-256, writes the tokenizer and the run config
`strata-lfm25-<size>.json`, and starts the model. `--model UD-Q3_K_XL` (or another size) skips the question, and
`--gguf-dir <folder>` uses a file you already have, under its name from the repository.

## How it differs from the other models

- **Most layers have no attention.** 30 of its 40 layers mix the tokens with a short convolution over the last
  three: each projects a token to three vectors B, C and x, convolves B times x, and scales the result by C. Its
  state is two vectors a layer. The other 10 layers attend over the whole conversation, with 8 key heads of 64
  values, so a context token takes 20,480 bytes of VRAM: a 128K context takes 2.0 GB more than 32K.
- **The experts.** 38 layers have 64 routed experts each, of which a token uses 4, with sigmoid gates, a bias that
  steers the pick and weights that sum to one. There is no shared expert. The first two layers have a plain FFN.
  The output matrix is the token table.
- **No draft block.** The files have nothing that guesses ahead, so the engine guesses only from text the
  conversation repeats, as for DeepSeek V4 Flash.
- **Tool calls** come as a Python list between two tokens, `<|tool_call_start|>[f(a='x'), g(b=1)]<|tool_call_end|>`,
  several calls in one; the server turns them into OpenAI's and Anthropic's tool calls.
- **Thinking.** The template always opens a `<think>` block; with thinking off the server closes it at once, as for
  GLM. Liquid AI's model card suggests temperature 0.1, top-k 50 and a repetition penalty of 1.05; setup does not
  set them. To use them for clients that send no sampling of their own, add
  `"sampling": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05}` to the run config.

## What is missing

- Pictures: the model reads text only.
- A draft model. Liquid AI publishes none for it.
- Contexts past 128K: it was trained for 131,072 tokens, and setup offers up to that.

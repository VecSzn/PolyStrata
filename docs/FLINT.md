# Flint 40B-A2.5B (`strata-poly`)

`strata-poly`, the engine of GLM 5.3 Flash and the smaller models, also runs Recursive's Flint 40B-A2.5B: llama.cpp's
architecture `flint`, read from Unsloth's GGUF files. The server, the web app and the API are the ones the README
describes.

Recursive trained Flint from scratch for coding agents: 39.8 billion parameters, of which a token uses about 2.5
billion ([its model card](https://huggingface.co/RecursiveSI/flint)). It is new here. This project has run it on two
machines: a Windows laptop whose card has 8 GB, and a Linux workstation with a 32 GB card. Its license is Apache 2.0.

## The sizes

| Size | From | Download | Experts | RAM beside an 8 GB card (setup's estimate) |
| --- | --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | [Unsloth](https://huggingface.co/unsloth/flint-GGUF): experts in IQ4_XS and Q5_K | 25.1 GB | 23.1 GB | about 29 GB |
| **UD-Q3_K_XL** | Unsloth: experts in IQ3_XXS and IQ3_S, IQ4_XS where they write | 18.5 GB | 16.5 GB | about 22 GB |
| **UD-Q6_K_XL** | Unsloth: experts in Q6_K, Q8_0 where they write | 36.3 GB | 34.2 GB | about 40 GB |
| **Q8_0** | Unsloth: 8-bit throughout | 42.3 GB | 40.2 GB | about 46 GB |

Setup takes UD-Q4_K_XL where the PC has about 15 GB of RAM past that estimate (about 40 GB beside an 8 GB card),
else UD-Q3_K_XL; the others are chosen by name (`--model Q8_0`). Every file is pinned to the repository's commit of
2026-10-21 and checked against its SHA-256.

## How fast

The README's numbers are PolyStrata's own benchmark through the server, the test every model of this engine gets
([BACKENDS.md](BACKENDS.md#a-prompt-of-30000-tokens)): `polystrata.py benchmark --prompt-tokens 24000`, a short chat
and a prompt built from this repository's documents, 256 tokens written after each, three runs a size and the last
two below, with the run configs setup writes for each machine.

### On a laptop with 8 GB of VRAM

An RTX 4070 Laptop with 8 GB, an i7-14700HX, 32 GB of RAM, Windows 10, CUDA 13.0. A context of 32,768 and 13 threads
for the experts the card does not hold, and a prompt read 2,048 tokens at a time. The prompt came to 29,165 tokens.

| Size | First token, s | Reads, tokens/s | Writes a short chat | Writes after the prompt | Experts from VRAM | Machine's RAM, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | 20.6 to 22.4 | 1,310 to 1,425 | 20 to 29 | 46 to 50 | 39% | 31.5 to 31.7 |
| UD-Q3_K_XL | 14.6 to 14.7 | 1,999 to 2,006 | 61 to 63 | 49 | 49% | 28.0 to 28.2 |
| UD-Q6_K_XL | 31.1 to 36.2 | 827 to 945 | 6 to 7 | 8 to 10 | 29% to 31% | 31.5 to 31.6 |
| Q8_0 | 37.2 to 38.3 | 767 to 790 | 3 to 4 | 5 to 6 | 26% | 31.5 to 31.6 |

Only UD-Q3_K_XL stays in the laptop's RAM beside the desktop's programs. The other three fill the 32 GB, and the
experts that drop out of the OS file cache are read from the SSD again while the model answers: UD-Q4_K_XL's short
chat came at 15 to 29 tokens/s over its three runs, and UD-Q6_K_XL and Q8_0 are of little use there. That is why
setup takes UD-Q4_K_XL only from about 40 GB of RAM beside an 8 GB card.

### On a workstation

An RTX 5090 (32 GB), 25 vCPU of a Xeon Platinum 8470Q, 90 GB of RAM for the container, Linux, CUDA 12.8. The card
holds every expert of UD-Q3_K_XL and UD-Q4_K_XL; of Q8_0 it holds 8,447 of the 12,032 at a 32K context, and the CPU
runs the rest. A prompt is read 2,048 tokens at a time. The prompt came to 28,827 tokens.

| Size | First token, s | Reads, tokens/s | Writes a short chat | Writes after the prompt | Experts from VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | 2.3 | 12,897 to 12,899 | 251 to 252 | 213 | 100% |
| UD-Q3_K_XL | 2.3 | 13,075 to 13,196 | 260 to 264 | 225 | 100% |
| Q8_0 | 4.3 to 4.4 | 6,737 to 6,969 | 158 to 160 | 140 to 166 | 88% to 89% |

UD-Q6_K_XL did not fit on the workstation's disk beside the other three, and was not run there.

### Against llama.cpp

On the workstation, `llama-bench` of llama.cpp a37cc46104 (the first with Flint), every layer on the card and flash
attention on, UD-Q4_K_XL; PolyStrata's engine with the same tokens, one at a time (`--window 1`):

| | PolyStrata | llama.cpp, batch of 512 (its default) | llama.cpp, batch of 2,048 |
| --- | ---: | ---: | ---: |
| Reads 29,816 tokens | 12,906 to 12,936 tokens/s (10,326 to 10,374 with a batch of 1,024) | 7,794 | 12,470 |
| Writes, one token at a time | 249 to 256 tokens/s | 222 | |
| Writes after 29,816 tokens, one at a time | 212 to 213 tokens/s | 188 | |

PolyStrata reads the prompt 3% to 4% faster than llama.cpp with a batch of 2,048 (11,843 in its first run after the
build) and 1.66 times as fast as with llama.cpp's default. One token at a time it writes 12% to 15% faster after a
short prompt and 13% faster after 29,816 tokens.

On the laptop, UD-Q3_K_XL through each server with the same prompts
([the runs](../bench/results/2026-10-24-laptop-flint-llamacpp-a37cc46/README.md)):

| | PolyStrata | llama.cpp a37cc46104, `--fit on` | llama.cpp, `--n-cpu-moe 36` |
| --- | ---: | ---: | ---: |
| Reads 3,566 tokens | 1,682 to 1,773 tokens/s | 402 to 406 | 392 to 409 |
| Writes after them | 63 to 69 tokens/s | 42.3 to 43.2 | 39.6 to 43.9 |
| Reads 29,442 tokens | 1,946 to 2,012 tokens/s | 390 to 391 | 379 to 394 |
| Writes after them | 49 to 52 tokens/s | 35.3 to 35.4 | 31.9 to 35.6 |

`--fit on` was the fastest of the three settings tried for llama.cpp there; `--n-cpu-moe 40` was slower still.

### What was tuned for it

UD-Q4_K_XL reading 29,816 tokens on the RTX 5090, tokens/s, and UD-Q3_K_XL's README pass on the laptop:

| Commit | Change | Effect |
| --- | --- | --- |
| 5ff592f1 | the model runs | 7,465 to 7,530 through the server, a prompt batch of 1,024 |
| c914fcf0 | setup's config reads 2,048 tokens at a time | 8,283 to 8,316; the laptop 1,155 instead of 840 to 901 |
| 372f4ece | a prompt batch's attention of heads of 128 values in registers (attn_flash128_kernel), and a few tokens' in blocks of 64 rows a launch block each | 11,864 to 12,025; one token at a time after 29,816 tokens 212 instead of 132 to 135 |
| 3863358e | the router's product in tiles of 4 tokens by 4 outputs a thread, the same bits | 12,929 to 12,936; the laptop 1,999 to 2,006 |

## Long contexts

Flint's model card gives it a context of 262,144 tokens; setup offers up to 131,072. The test is the one DEEPSEEK.md
has: a document of this repository's docs with three facts planted at 15%, 50% and 85% of it, then three questions.
On the workstation, UD-Q4_K_XL with a context of 131,072 read and answered the 124,122-token request through the
server in 20.4 s and found all three facts (39.6 s before 372f4ece). Flint always thinks, and its thinking quoted
the three sentences before it answered; it wrote 170 tokens in all.

On the 8 GB laptop, PolyStrata's benchmark with the context raised in setup's run config, twice each:

| Size | Context | Prompt | First token, s | Reads, tokens/s | Writes after it |
| --- | ---: | ---: | ---: | ---: | ---: |
| UD-Q3_K_XL | 65,536 | 62,744 tokens | 39.7 to 40.2 | 1,567 to 1,585 | 37 to 38 |
| UD-Q3_K_XL | 131,072 | 126,188 tokens | 118 to 128 | 987 to 1,069 | 23.5 to 24.7 |
| UD-Q4_K_XL | 65,536 | 62,744 tokens | 47.3 to 47.5 | 1,326 to 1,332 | 21 to 28 |
| UD-Q4_K_XL | 131,072 | 126,188 tokens | 126 to 133 | 953 to 1,007 | 17 to 21 |

A 128K context takes 3.5 GiB of the card's 8 for its rows, so nearly every expert runs on the CPU there; UD-Q4_K_XL's
file also does not stay in RAM. Setup's context menu shows UD-Q3_K_XL's numbers.

## How it was checked

- The tokenizer gives the same ids as llama.cpp's `llama-tokenize` on 195,861 tokens: three READMEs (English,
  Chinese, Japanese), BACKENDS.md, a Python file and a CUDA file.
- With greedy sampling and the expert cache fixed, the engine and llama.cpp with its experts on the CPU write the
  same tokens on two short prompts, 34 of 34 and 64 of 64, on the laptop and on the workstation. After the first 100,
  600, 1,500 and 3,740 tokens of a longer prompt they agree on 11, 24, 6 and 24 of 24 tokens on the workstation. The
  1,500-token prefix parts where llama.cpp gives the two most likely tokens 0.400 and 0.205 and the engine 0.287 and
  0.402; read token by token with its plain float kernels the engine gives them 0.242 and 0.258.
- Three answers written together are the ones written one at a time, all 64 tokens of each, after a prompt of 15
  tokens and one of 3,000, with every expert on the card and with the card held to 8 GB.
- `poly_sink_attend_test` checks the attention of a prompt batch (attn_flash128_kernel) and of a few tokens in
  blocks of rows (attn_split_kernel) against the plain kernels, with the sinks and a window of 513 over a ring of
  rows; `serve/test_flint.py` checks the pre-tokenizer, the tool calls, the stop tokens and the thinking.

## How the sizes compare

Q8_0 wrote 256 greedy tokens on each of five prompts (code, a summary of 4,987 tokens of BACKENDS.md, arithmetic,
a story, a question in Chinese); each other size then followed those tokens, and the share of steps where its most
likely token is Q8_0's was counted. Q8_0 stands in for the model as trained here. UD-Q4_K_XL and UD-Q3_K_XL were
compared on the workstation; UD-Q6_K_XL, which did not fit there, on the laptop against the laptop's own run of Q8_0.

| Size | Same most likely token as Q8_0 | Mean probability of Q8_0's token |
| --- | ---: | ---: |
| UD-Q6_K_XL | 97.8% (1,252 of 1,280) | 0.870 |
| UD-Q4_K_XL | 95.9% (1,228 of 1,280) | 0.854 |
| UD-Q3_K_XL | 94.0% (1,203 of 1,280) | 0.844 |

Q8_0 gives its own tokens 0.861 on average on the workstation and 0.872 on the laptop.

## What you need

- An NVIDIA card, RTX 20 or newer, with 8 GB of VRAM or more. About 4 GB go to the weights outside the experts, the
  caches and the buffers at a 32K context; the rest holds experts. A 32 GB card holds UD-Q4_K_XL whole.
- RAM for the experts the card does not hold. They are read from the mapped file while the model answers, so they
  should stay in the OS file cache: setup's estimate is in the table above. On the 32 GB laptop only UD-Q3_K_XL
  stayed there beside the desktop's programs.
- Linux or Windows 10 / 11. AMD cards: the engine compiles for them on Linux and under Windows as well (AMD's TheRock
  wheels); this model has not been tried on one, and setup says so.

## Install

```
./setup.sh --family flint
```

On Windows, `START-HERE.bat` and pick it from the list. Setup asks for the size, the context and whether to keep
several conversations in RAM, downloads the file, checks its SHA-256, writes the tokenizer and the run config
`strata-flint-<size>.json`, and starts the model. `--model UD-Q3_K_XL` (or another size) skips the question, and
`--gguf-dir <folder>` uses a file you already have, under its name from the repository.

## How it differs from the other models

- **Most layers look back 513 tokens.** 35 of its 49 layers attend to the current token and the 512 before it, with
  RoPE; their keys and values are a ring of rows that does not grow with the conversation. The other 14 attend over
  the whole conversation and have no position encoding at all. 4 key heads of 128 values serve the 32 query heads,
  so a context token takes 28,672 bytes of VRAM: a 128K context takes 2.6 GB more than 32K.
- **Attention sinks.** Each head has a learned value that joins its softmax like the score of one more key with no
  value, so a head can give most of its weight to nothing. The engine adds it to the softmax of a token and of a
  prompt batch.
- **The experts.** 47 layers have 256 routed experts each, of which a token uses 6, with sigmoid gates and weights
  that sum to one, and a shared expert that every token uses. The first two layers have a plain FFN. The output matrix
  is its own, not the token table.
- **Large activations.** In layer 46 the input of the down experts reaches 45,000, and a block of 32 of them sums to
  92,660, past the largest half-precision number (65,504). The engine reads a prompt batch with llama.cpp's matrix
  kernels, which keep that sum in half precision for Q4_K and Q5_K, so it became infinite and the answer a row of
  token 0. For Flint the engine scales such a row down by a power of two before it is quantized and back up after the
  product, which changes no bit of the result. The other models' sums stay in range, and they skip the check: it
  cost LFM2.5 4% of a 30K prompt's read.
- **No draft block.** The files have nothing that guesses ahead, so the engine guesses only from text the
  conversation repeats, as for LFM2.5 and DeepSeek V4 Flash.
- **Tool calls** come as GLM's: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`. The server
  turns them into OpenAI's and Anthropic's tool calls, and stops an answer at `<|user|>`, `<|observation|>` and
  `<|developer|>` as well as at the end token, as llama.cpp does.
- **Thinking.** The model opens its `<think>` block itself and thinks whatever a request says: its template has no
  switch for it. The server returns the thinking as `reasoning_content`; Recursive asks clients to send it back in
  the history, and the template puts it there. The model card suggests temperature 1.0, top-p 0.95 and top-k 20;
  setup does not set them. To use them for clients that send no sampling of their own, add
  `"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}` to the run config.

## What is missing

- Pictures: the model reads text only.
- A draft model. Recursive publishes none.
- Contexts past 128K in setup, although the model card gives 262,144: not tried here.

# GLM 5.3 Flash (`strata-poly`)

PolyStrata adds a second engine, `strata-poly`, that runs GLM 5.3 Flash. It sits beside `strata`, which still runs
Qwen3.8-Flash-Next as before. The server, the web app and the API are the same for both: a run config names the
engine it starts.

It is early. This project has built and run it on one machine, with Linux and one NVIDIA card, and on two RTX
3080s; a user runs it on Windows with a 24 GB card and 128 GB of RAM ([issue 10](https://github.com/VecSzn/PolyStrata/issues/10)).

## What it was measured on

| | |
| --- | --- |
| Graphics card | RTX 5090, 32 GB |
| CPU | Xeon Platinum 8470Q, 25 vCPU of it (a cloud machine), 24 threads used |
| RAM | 120 GiB |
| Built with | CUDA 12.8 |
| Model | `GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf`, 113.6 GB, from [neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) |
| Context | 32K |

| | `strata-poly` | llama.cpp ([neurall's fork](https://github.com/neurall/llama.cpp)), stock settings |
| --- | ---: | ---: |
| Writes answers | 46 tokens/s after the prompt of 11,673 tokens and 62 to 66 after the one of 29,069, 62 to 65 for a 400-token answer to a short one | 26 tokens/s |
| Reads a prompt of 3,059 tokens | 641 to 648 tokens/s | about 265 tokens/s |
| Reads a prompt of 11,673 tokens | 760 to 817 tokens/s | |
| Reads a prompt of 29,069 tokens | 740 to 782 tokens/s | |
| Starts | about 7 s with the file in the page cache | |

Both ran on the machine above. Nothing here has been run on a smaller card.

The engine's numbers are two runs of the engine of 2ba29a383baf, with what setup gives it: the draft block, and
the expert profile the repository ships after one run has added its counts. A cell has the higher of these and
two runs of the build of 2026-10-06, which read the three prompts at 610 to 620, 788 to 802 and 733 to 742 tokens/s
and wrote 47 to 51 after the long ones; the 400-token answer is that build's, and came at 64 to 65 from this one.
The 29,069-token prompt starts like the 11,673-token one before it, so its first 10,240 tokens are not read
again. An earlier build, measured the same way without the draft block, wrote 26.7 to 27.7 tokens/s and read the
three prompts at 373, 334 and 312; [BACKENDS.md](BACKENDS.md#what-was-tuned) has what changed since. llama.cpp was
not run again.

The README's row for 3.0BIT is PolyStrata's own benchmark through the server, the test every model of this engine
gets ([BACKENDS.md](BACKENDS.md#a-prompt-of-30000-tokens)), with the engine of the ready-made release: a short chat
written at 59 to 66 tokens/s, and a prompt of 29,154 tokens read whole at 820 to 824, with the first token after
36 s and the answer after it written at 62 to 64. The build before read a prompt of 28,354 tokens there at 784 to
785, and the engine of 2ba29a383baf one of 29,364 at 786 to 810, with the first token after 36.3 to 37.5 s, and
wrote the short chat at 60 to 64 and the answer after the prompt at 61 to 62. Of the 29,069-token prompt in the
table above only the last 18,829 tokens are read, the ones furthest into the context, which is the likely reason
it comes out slower.

With greedy sampling the two write the same tokens until two candidates are nearly tied. The model's activations
are quantized to 8 bits between layers, so a difference in rounding between two correct implementations can decide
such a tie.

## What you need

- Linux and an NVIDIA card, RTX 20 or newer. It has only run on 32 GB of VRAM. About 12 GB go to the weights, the
  caches and the buffers before any expert, so setup stops and asks below 14 GB.
- RAM for the experts the card does not hold. The file is mapped and they are read from it while the model answers,
  so they should stay in the OS file cache: about 86 GB of the 107 GB on a 32 GB card. Setup's estimate for that card
  is 96 GB of RAM. It has run with 120 GB only. In a container, the memory limit must leave about 8 GB above the
  model's files: with less, answers read experts from the disk, and the container may end the engine
  ([MULTI_GPU.md](MULTI_GPU.md#the-strata-poly-models) has the pod it happened on). The engine says so at load.
- 114 GB of disk for the model file, and 3 GB for its draft block.

## Sizes

| Size | What it is | Download | RAM beside a 32 GB card |
| --- | --- | ---: | ---: |
| `3.0BIT` | The whole model, 288 routed experts a layer: the 3-bit file from [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF). | 114 GB | 96 GB |
| `UD-IQ2_XXS` | The whole model at about 2 bits: [Unsloth's](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) quantization, in four files. Its experts are i-quants, which pass through the card faster and through the CPU more slowly than K-quants: it reads a prompt faster than `3.0BIT` and writes more slowly. | 102 GB | 82 GB |
| `REAP50-Q3_K_M` | [REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) by patrickbdevaney: half of each layer's routed experts removed, the rest as 3-bit K-quants. Its card says what was removed and how it was checked. | 79 GB | 60 GB |

The RAM is setup's estimate: the experts the card does not hold, and 10 GB beside them. Without `--model` setup
takes the first size whose estimate the PC meets.

The other two sizes on the machine above, measured like the table at the top, twice each with the engine of
2ba29a383baf. Both files carry their own draft block.

REAP50-Q3_K_M writes a 400-token answer at 68 to 69 tokens/s, 52 to 55 after the prompt of 11,673 tokens and 71
to 75 after the one of 29,069, and reads prompts of 3,059, 11,673 and 29,069 tokens at 1,038 to 1,043, 1,279 to
1,285 and 1,253 to 1,257 tokens/s. The card holds 21.6 GiB of its experts, 1,939 of 6,192. By the README's test it
reads 29,364 tokens at 1,289 to 1,292 tokens/s, with the first token after 22.8 to 22.9 s, and writes a short chat
at 62. Against llama-server of the llama.cpp fork the file was made for, on a prompt of 1,678 tokens, the first 63
tokens were the same; that was an earlier build, and it was not compared again.

UD-IQ2_XXS writes the 400-token answer at 49 tokens/s, 39 after the prompt of 11,673 tokens and 54 to 57 after
the one of 29,069, and reads the three prompts at 805 to 812, 1,020 to 1,030 and 1,014 to 1,021. The card holds
19.6 GiB of its experts, 2,768 to 2,771 of 12,384. By the README's test, in three runs in turn with the build
before, it reads 29,364 tokens at 1,062 to 1,076 tokens/s, with the first token after 27.4 to 27.8 s, and writes a
short chat at 50 to 51.

The build of 2026-10-06 wrote the 400-token answer at 61 tokens/s with REAP50-Q3_K_M and at 44 to 45 with
UD-IQ2_XXS, and read the 29,069-token prompt at 1,207 to 1,209 and 973 to 983. An earlier build read it at 384
tokens/s with REAP50-Q3_K_M and at 347 with UD-IQ2_XXS, and wrote 27 and 33.

## Install

```
./setup.sh --family glm
```

Setup checks the PC and asks three things: the size, the context, and whether to keep several conversations in
RAM. Then it
compiles the engine, downloads the model, checks its SHA-256 and starts it. For 3.0BIT it also downloads the model's
draft block, 2.8 GB, once for every size that needs it. The web app is at
`http://127.0.0.1:8080`. After that, `./setup.sh` starts the model again without asking.

- The compile needs g++ and the CUDA toolkit (12.8 or newer for an RTX 50 card). Setup offers to install what is
  missing. The ready-made `strata-poly` of a release is for Windows only.
- You already have the GGUF: `./setup.sh --family glm --gguf-dir /folder/with/the/file`.
- The other options are the README's: `--context`, `--port`, `--host`, `--api-key`, `--gpu`, `--gpus`,
  `--layer-split`, `--yes`, `--no-start`.
  The ones for the Qwen engine (`--kv`, `--vision`, `--low-ram`, `--parallel`, ...) are named and left out.

On the test machine, with the file already on the disk, this took about five minutes.

Setup writes the run config `strata-glm-3.0bit.json` and `run-glm-3.0bit.sh` into the PolyStrata folder, the engine into
`engine-poly/`, and the tokenizer and the expert profile into the data folder. The profile starts as the counts in
`data/expert-profile-glm.txt`, which come from the test runs here, and the engine updates it after an answer, every
10 seconds at most, and when it stops.

A 64K or 128K context starts and answers, with an expert cache 0.8 or 2.6 GiB smaller. No long prompt has been read
at those sizes.

## Without setup

Build the engine:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
cmake --build build --target strata-poly
```

Write the tokenizer out of the GGUF once:

```
python tools/strata_tokenizer.py --gguf /models/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf --out glm-tokenizer
```

Then write a run config, `glm.json`:

```json
{
  "exe": "build/strata-poly",
  "cwd": "/path/to/PolyStrata",
  "args": ["--model", "/models/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf",
           "--max-context", "32768",
           "--expert-profile", "glm.profile",
           "--conversation-cache-mib", "8192"],
  "tokenizer": "glm-tokenizer/tokenizer",
  "model_name": "glm-5.3-flash",
  "log": "glm-engine.log"
}
```

and start the server with it:

```
python serve/server.py --engine strata --config glm.json --port 8095
```

The web app is at `http://127.0.0.1:8095`, and the API is the one the README describes. Chat, thinking, tool
calls and streaming were checked through `/v1/chat/completions`. `"enable_thinking": false` in
`chat_template_kwargs` turns thinking off.

## How it uses the card

The model has 45 layers. 42 of them choose 8 of 288 experts for every token, which makes 12,096 experts and most
of the file. The rest of the weights are always in VRAM.

- **The expert cache.** The experts used most often are kept in VRAM: about 2,470 of them, 19.2 GiB, on the 32 GB
  card at a 32K context with the draft block. `--expert-profile FILE` is where the engine keeps count of which ones are used. It reads the
  file at the start to choose, and writes it back when it stops. Without a profile it has no counts yet and keeps
  the same number of experts from every layer, so the second start is the one to measure.
- **The cache follows the answer.** The choice at the start fits the average answer, and some answers badly: of
  the expert uses of a 400-token answer with code in it, 20% found their expert in VRAM. So while an answer is
  written, the engine counts which experts the last few hundred tokens used. The ones they keep choosing are copied
  into VRAM, each over a cached expert they chose less than half as often, up to 16 on their way at a time. For
  that answer the share went to 50% and the speed from 26.4 to 30.7 tokens/s. The cache stays as the last answer
  left it, so the next one starts from there. A prompt does not move it. `--expert-swaps 0` leaves the cache as it
  was loaded, and the same prompt then gets the same answer from the second request on: an expert's result
  differs in its last digits between the CPU and the card, and with swaps an expert can run on one in one run and
  on the other in the next. The first request can still differ, since the cache is sized once more before the
  second. A prompt of 32 tokens or more needs `--cpu-share 0` beside it: the CPU's share of such a call's experts
  follows the calls before.
- **Writing an answer.** An expert that is not in VRAM runs on the CPU while the card runs its share. The CPU reads
  such an expert from memory, and on a PC with more than one NUMA node that memory is on the node whose CPU first
  read the file. The engine looks where the experts' pages are and keeps its CPU threads on that node: from the
  other node the same work took 30% longer on the machine above. `--any-node` turns that off.
- **Guessing ahead.** While an answer is written, the engine guesses the next few tokens and puts the token it
  knows and up to three guesses through the model together. A guess counts only when it is the token the model
  picks itself, so the answer is the same with guessing and without. The guesses come from two places. An answer
  often repeats text the conversation already has, code it was asked to change or a passage it quotes: when the
  last few tokens match an earlier place, what followed there is the guess. And the model has a block made for
  guessing, one more layer that takes a token and what the model made of the token before it, and says what is
  likely to come next. The block is the model's, one for every size. Unsloth's files carry it. The 3.0BIT file
  leaves it out, so setup downloads the block alone for it, 2.8 GB out of Unsloth's files, and gives it to the
  engine as `--draft-model`. REAP50 has its own, pruned like its layers. A right guess saves less than a whole
  token, because its experts outside VRAM still run on the CPU: four tokens together take about three times as
  long as one. Measured on the 32 GB card with an earlier build, 600-token answers, twice each, in tokens/s:

  | Answer | 3.0BIT without guesses | with | REAP50-Q3_K_M without | with |
  | --- | ---: | ---: | ---: | ---: |
  | plain text | 40 | 46 to 47 | 41 | 44 to 46 |
  | quotes from a document | 42 | 49 to 51 | 41 | 48 to 51 |
  | a 670-token file given back with one name changed | 42 | 51 to 52 | 42 | 50 to 59 |

  Of 3.0BIT's guesses 87 to 98% were right. UD-IQ2_XXS gains little: its experts are i-quants, which the CPU
  computes more slowly, so a guessed token costs it nearly what a token costs. Three 200-token answers wrote 30.7,
  33.7 and 31.4 tokens/s with its block and 31.3, 30.9 and 31.4 without. `--no-draft` and `--lookup 0` turn the two
  sources off.
- **Reading a prompt.** The prompt goes in up to 1,024 tokens at a time. For each layer the experts outside the
  cache are copied through a buffer in VRAM and run there. A larger batch reads faster and leaves less VRAM for the
  expert cache, so answers get slower: with an earlier build, which read the 11,673-token prompt at 334 tokens/s
  and wrote about 27, 4,096 at a time read it at 396 and wrote 18.5.

## Several conversations at once

The engine holds one conversation. A coding agent rarely sends just one: subagents and short side requests come in
between the turns of the main one. Each switch used to make the engine read the whole prompt again.

`--conversation-cache-mib N` gives the engine N MiB of RAM to park conversations in. When another conversation
needs the engine, the current one is copied to RAM. When it comes back it is copied in again and goes on where it
stopped. Two conversations that start with the same text, a shared system prompt for one, share what is saved for
that part.

Measured through the server with 8,192 MiB: a main conversation of 3,884 prompt tokens, a subagent of 3,323 and a
short request took turns. Every later turn of the first two had 97.8 to 99.2% of its prompt cached and took 1.3 to
3.9 s. The first turns, which read the prompt, took 13 s.

What it costs: a checkpoint is 153 MB of RAM, and a parked conversation takes 28 KB for each token. At the end of
that run two parked conversations held 1.1 GB.

The API reports what was not read again as `usage.prompt_tokens_details.cached_tokens` (Responses:
`input_tokens_details.cached_tokens`, Messages: `cache_read_input_tokens`), and the web app's Monitor has a
Conversation cache card.

It is off by default, as it is in `strata`. Without it the engine still keeps the last conversation: a next turn
goes on from where it stopped, and an edit goes back to the nearest checkpoint.

## Options

| Option | Default | What it does |
| --- | ---: | --- |
| `--model FILE.gguf` | | the model |
| `--serve` | | take requests from the server (the server passes this) |
| `--max-context N` | 8192 | tokens a conversation can hold |
| `--threads N` | all usable | CPU threads for the experts that are not in VRAM |
| `--vram-reserve-mib N` | a thirty-second of the card, 256 to 768, or a sixteenth, 512 to 1024, kept whole, when the PC's free RAM is within 4 GiB of the model's files; else half of it after the first request | VRAM left free beside the expert cache |
| `--layer-split S` | the first card alone | the model's layers across the cards the engine sees: `auto` (by each card's free VRAM) or the first layer of each later card, as `20` or `20,30` ([MULTI_GPU.md](MULTI_GPU.md#the-strata-poly-models)) |
| `--expert-profile FILE` | | expert use counts, read at the start and written back after an answer, every 10 seconds at most, and at the end |
| `--no-gpu-experts` | | keep every routed expert on the CPU |
| `--any-node` | | on a PC with several NUMA nodes, let the CPU threads run on any of them |
| `--no-graphs` | | launch a token's kernels one by one, not as the CUDA graph a layer that is recorded at the first token |
| `--expert-swaps N` | 64 | while an answer is written, experts on their way into VRAM at a time, each in place of a cached one the answer uses less; 0 = the cache stays as it was loaded |
| `--window N` | 4; 8 for the other models | the most tokens that go through the model together while an answer is written: the known one and guesses after it: the draft block's three at most, and more than three only from a repeat of eight tokens or more; 1 = no guesses |
| `--lookup N` | 3 | guess what followed an earlier repeat of the last N or more tokens; 0 = never |
| `--no-draft` | | do not run the block that guesses ahead |
| `--draft-model FILE.gguf` | | a file of the same model to take that block from, when `--model` has none; setup passes the one it downloaded |
| `--draft-vocab N` | 131072 for Qwen3.6, Ornith and Tiel Coder, all of it for the others | the block guesses among the vocabulary's first N tokens; 0 = all |
| `--kv-whole` | on an AMD card | give the attention caches VRAM for the whole context at load, not as the conversation grows; setup writes it for an AMD card, where ROCm keeps the VRAM the expert cache gives back ([AMD_HIP.md](AMD_HIP.md#strata-poly-on-amd-beta)) |
| `--prompt-batch N` | 1024 | prompt tokens read at a time, 1 to 4096; about 1.4 MiB of VRAM each |
| `--stage-mib N` | 2048 | VRAM the experts outside the cache pass through while a prompt is read |
| `--stage-ahead N` | 1024 | from N tokens, a batch copies those experts before the layer has chosen, as does every batch of a prompt of N tokens or more that is read several at a time; 0 = never |
| `--cpu-share F` | as the times say | in a batch of fewer tokens than `--stage-ahead`, the share of the experts outside VRAM that the CPU computes while the others are copied: the ones the fewest tokens chose; 0 = none |
| `--prompt-cache N` | 6 | checkpoints kept of a conversation; 0 = only a next turn is not read again |
| `--prompt-cache-every N` | 2048 | a checkpoint every N prompt tokens, besides the one at a prompt's end; 0 = none |
| `--conversation-cache-mib N` | 0 | RAM for parked conversations; 0 = off |
| `--conversation-cache-slots N` | 4 | parked conversations at most |
| `--conversation-cache-min-tokens N` | 1024 | a shorter conversation is not parked |
| `--conversation-cache-min-free-mib N` | 2560 | park only while this much physical RAM stays free |
| `--eos-ids ID,ID,...` | from the GGUF | tokens that end a reply |
| `--tail-skip N` | 0; setup gives Qwen3.6 6 | an expert outside VRAM that every token of a window ranks at place N+1 or later among its experts is left out: faster, and the answers move a little |

`strata-poly` with no arguments prints the list, with the options for running it without the server.

## What is missing

- Other files of the model. The engine reads the layouts of four converters (neurall's llama.cpp fork,
  patrickbdevaney's, antirez's and Unsloth's), a model in several files, and the weight types llama.cpp's kernels
  have. Only the three sizes above have been run: setup offers no other.
- A ready-made `strata-poly` off Windows, or for a driver older than CUDA 13's: setup compiles it there.
- GLM on Windows, by this project: no PC with the RAM for it was at hand, and setup asks before it installs it
  there. A user ran 3.0BIT on Windows 10 with an RTX 5090 D v2 (24 GB) and 128 GB of RAM
  ([issue 10](https://github.com/VecSzn/PolyStrata/issues/10)): 1,483 experts in VRAM, 26 to 30 tokens/s after
  a short prompt, 10K tokens read at 375 to 395. The context set to 1,048,576 in the run config loaded and
  answered, and one request of 557K tokens was read at 286 tokens/s with the three facts planted in it found.
  With 0.1.44.1-poly.5 a request of 1,002,452 tokens was read in 45.7 minutes (66.3 with poly.2), the three
  facts found, and from the second request after it the engine wrote at 25 to 30 tokens/s again.
- Intel cards, and AMD cards on Windows. On Linux setup compiles the engine for an AMD card
  ([AMD_HIP.md](AMD_HIP.md#strata-poly-on-amd-beta)), a beta tested with Qwen3.6 and Ornith on an RX 7600; GLM
  itself has not been tried on one, and setup says so. Two NVIDIA cards can share the model
  ([MULTI_GPU.md](MULTI_GPU.md#the-strata-poly-models)): REAP50 has run on two 10 GB cards under Linux.
- Pictures. The model's image input is not read.
- Several requests decoded together (`"parallel"`). Requests wait their turn.
- `--prompt-cache-root` and `--turn-token`. Checkpoints are taken by position.

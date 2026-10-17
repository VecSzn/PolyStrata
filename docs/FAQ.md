# Questions people ask

Answers to what people asked about PolyStrata when it was first posted. The numbers are measured on the machines
named next to them; how is in [BACKENDS.md](BACKENDS.md).

### Why not contribute this to Strata instead of a separate repo?

Someone asked Strata for GLM and DeepSeek in [issue 310](https://github.com/Niko1221/Strata/issues/310), and the
maintainer closed it: "Other MoE architectures (DeepSeek, GLM) aren't planned: the kernels and the pack format are
built around this model." So the other models got an engine of their own, next to Strata's.

The repo keeps Strata's git history and merges its releases, and Strata's own engine is in it unchanged. There are
now open PRs in Strata for GLM 5.3 Flash (#1315) and DeepSeek V4 Flash (#1518). If Strata starts taking other
architectures, I'd like this work to go there, and I'm glad to help with that.

### Is the second engine related to Strata at all?

Yes. It is built on Strata's own code: its quantized matvec kernels, its GGUF reader, its draft policy and suffix
drafter, its conversation memory and the llama.cpp MMQ kernels Strata compiles. You can see the list in
`CMakeLists.txt` (the `strata-poly` target). What is new is the model code for the five architectures and the
part that decides which experts sit in VRAM and which run on the CPU.

### How do I know the output is right?

It reads the same GGUF files llama.cpp reads, and its output is compared with llama.cpp's on the same file: greedy
decoding, 64 tokens after five chat prompts, token by token. On the laptop, Qwen3.6-35B-A3B UD-Q3_K_M matched all
64 tokens on all five prompts. The other sizes and models, with the few places where they part ways and why, are
in [docs/BACKENDS.md](BACKENDS.md#how-each-backend-was-checked).
Guessing tokens ahead does not change the answer: with every expert on the CPU, the tokens are the same with and
without guesses.
Those checks left out no experts. Setup starts Qwen3.6 with `--tail-skip 6`, which skips a few low-ranked
experts that are not in VRAM and moves its answers a little; `STRATA_ROUTE_TAIL_SKIP=0` turns that off
([measured](BACKENDS.md#what-was-tuned)).

Every speed in the README says which machine, which file and which prompt, and the benchmark that produced it is in
the repo, so you can run it on your own PC.

### How fast will it be on my card?

The README's numbers are from two machines: a laptop with an 8 GB RTX 4070 and a workstation with an RTX 5090.
What people measured on their own cards is in the
[community results](COMMUNITY_BENCHMARKS.md#the-models-polystrata-adds): so far an RTX 4070 SUPER and an
RTX 4070 Ti with 12 GB, and a laptop's RTX 5060 with 8 GB. Yours can go there too; the page says what to send.

### How can I compare it with llama.cpp myself?

With `bench/bench_vs_llama.py`, the script Strata used for its own comparison. It runs PolyStrata and llama.cpp's
`llama-server` in turns on the same GGUF and the same prompts and prints the medians of both
([how to run it](COMMUNITY_BENCHMARKS.md#comparing-with-llamacpp)). On the README's 8 GB laptop, with Qwen3.6
and with Tiel Coder, PolyStrata read the prompts 2.7 to 3.5 times as fast as llama.cpp b11540 and wrote 1.5 to 1.8
times as fast (the two runs are linked there).

### Does it run on Radeon / AMD?

Qwen3.8-Flash-Next, yes: on a Radeon with 12 GB or more it runs through Strata's AMD backend. The five added
models run on AMD under Linux since 0.1.42-poly.4, as a beta: setup compiles their engine for the card on your PC
(there is no ready-made AMD engine, so not on Windows), a contributor's port tested with Qwen3.6 UD-Q2_K_XL and
Ornith APEX-I-Mini on an RX 7600 with 8 GB: 74 and 40 tokens/s in a short chat, 53 and 43 after a 32K prompt,
read at 481 and 444. Gemma 4, Tiel Coder, LFM2.5 and GLM have not been tried on an AMD card, and setup says so when
you pick them. I don't have an AMD card myself, so reports from one help; see
[AMD_HIP.md](AMD_HIP.md#strata-poly-on-amd-beta).

### Will it run on a 6 GB card, or 4 GB?

It has only run on 8 GB cards and up. With a context of 32K, Qwen3.6 takes about 5 GB of VRAM beside its experts,
Ornith about 4 GB, Gemma 4 about 6 GB, Tiel Coder about 5 GB and LFM2.5 about 4 GB. The attention caches are part
of that and take their share as a conversation grows, so a short chat leaves more room. On a 6 GB card Ornith and
LFM2.5 are the ones to try; LFM2.5's caches grow the least, 20 KB a token.
That is untested. On 4 GB, none of them leaves room for experts.

### What about older cards like a GTX 1050?

The new engine needs an RTX 20 series card or newer. GTX 10 series cards are not supported by it. Strata has an
experimental build for older cards, but that is for Qwen3.8-Flash-Next only
([docs/OLDER_GPUS.md](OLDER_GPUS.md)).

### Can you add Bonsai, LFM2.5, Ling or DeepSeek V4 Flash?

This engine helps MoE models that don't fit in VRAM. A dense model like Bonsai uses all its weights for every
token, so there are no experts to split, and the 1-bit Bonsai models already fit on a small card. LFM2.5-8B-A1B
fits whole on an 8 GB card, so plain llama.cpp is fine there. LFM2.5-24B-A2B, the larger one, runs now
(`--family lfm25`, [LFM.md](LFM.md)). Bigger MoEs are the good fit. DeepSeek V4 Flash runs now (`--family
deepseek`, [DEEPSEEK.md](DEEPSEEK.md)), and
Ling-2.6-flash (104B, 7.4B active) is on the list. Requests go in the issues, and the ones people ask for most come
first.

### Can I run a fine-tune, or another model with the same architecture?

People have, on their own PCs. On Reddit, one ran HauhauCS's uncensored Qwen3.6-35B-A3B at Q4_K_M, which was not
tried here, and another Tiel Coder 35B-A3B, a quantization of Ornith 1.5 that setup now installs (`--family tiel`).
Their numbers are in the
[community results](COMMUNITY_BENCHMARKS.md#the-models-polystrata-adds). The engine reads the architecture and the
weights from the GGUF, so a fine-tune that keeps the architecture of Qwen3.6, Ornith 1.5 or Gemma 4 should run.
Setup only knows the files it offers. To give it another, put the file in a folder under the name of the size
closest to it, `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` for a Q4_K_M of Qwen3.6, and run setup with
`--family qwen36 --model UD-Q4_K_M --gguf-dir <that folder>`.

### Can I use any GLM 5.3 Flash GGUF?

The installer offers three files, and those are the ones that were run: neuralll's 3.0bit (114 GB), Unsloth's
UD-IQ2_XXS (102 GB) and patrickbdevaney's REAP50-Q3_K_M (79 GB, half of the routed experts removed). Other GGUFs
from the same converters should load, but they haven't been run. If you already have one:
`./setup.sh --family glm --gguf-dir /folder/with/the/file`.

GLM needs a lot of RAM: beside a 32 GB card, 96 GB for the 3.0bit file, 82 GB for UD-IQ2_XXS and 60 GB for REAP50.
So far it has run on Linux only.

### Why Q3 and Q2 by default on the smaller models? Isn't that a big quality loss?

A 4-bit size is there for all four smaller models: `--model UD-Q4_K_M`, for Ornith `APEX-I-QUALITY` and for Tiel
Coder `UD-Q4_K_XL`. The installer picks UD-Q3_K_M because on an 8 GB card more of its experts fit and its file is
lighter on 32 GB of RAM. The speeds are close on the laptop: Qwen3.6 writes a short chat at 94 to 100 tokens/s
with Q3 and 84 to 85 with Q4. It only takes Q2 when the RAM is too small for Q3. On a card that holds the whole
model, Q3 and Q4 run at about the same speed, so take Q4 if you have the memory.

### What about NVFP4?

PolyStrata reads GGUF weight types only, so no NVFP4 for now. For GLM, NVFP4 would make the model around 204 GB,
which doesn't fit in RAM on a normal PC; the 2- and 3-bit GGUFs (79 to 114 GB) are what lets the experts stay in RAM.

### Does it work on Windows?

The four smaller models (Qwen3.6, Ornith 1.5, Gemma 4, Tiel Coder) and Qwen3.8-Flash-Next do. Setup downloads the
new engine ready-made from the latest release, built for RTX 20 to RTX 50 cards. On a processor without AVX2, or
when the engine's code changed after that release, it compiles the engine instead with Visual Studio's build tools
and the CUDA toolkit, and offers to install them. GLM 5.3 Flash has only run on Linux.

### Can I use it with Claude Code, Codex or other apps?

Yes. It serves OpenAI- and Anthropic-compatible APIs on localhost, with tool calls. Claude Code:
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`. Codex CLI uses the Responses API at `/v1/responses`
([setup](DETAILS.md#the-responses-api-and-codex-cli)).

For Claude Code, also add `"anthropic_thinking": "on_request"` and `"fit_max_tokens": true` to
`strata-<model>.json` and restart. Claude Code first asks for a session title, and without the first setting the
model thinks that over instead, and runs out of the 512 tokens the request allows before it has a title: 5 s a
request with Qwen3.6 on an 8 GB laptop ([more](DETAILS.md#using-it)). The second one it needs with the 32K context
setup suggests: Claude Code asks for
32,000 tokens of room for the answer in every request, and the server refuses a request when the prompt leaves less
than that. With the setting the server gives it the room that is left. With `"warm_start": true` as well, the server
keeps Claude Code's system prompt and tool list and reads them when it starts, so the first request after a restart
reads only the rest: 3.5 s instead of 11.1 s in a test.

### A long chat stops with "request exceeds the context window". Can it drop old messages instead?

Yes, if you ask for it. By default the server never cuts a conversation: when it no longer fits the context, the
request gets that error and the engine keeps running. Add `"drop_oldest": true` to `strata-<model>.json` and restart,
and the server leaves out the oldest turns (a question with its answers and tool results) until the prompt leaves
room for the answer, as LM Studio's rolling window does. The system prompt and the last message stay, and the server
window says how many messages it left out. Each request is cut again from the full conversation the app sends, so
the model has forgotten what was dropped. Claude Code, Codex and OpenCode compact the conversation themselves and do
not need it. A larger context (`--max-context` in the same file) keeps more of the chat.

### Can the model see what it thought in earlier turns?

With Qwen3.6, Ornith and Tiel Coder, if you ask for it. Their template leaves the thinking of earlier answers out of
the prompt, so a model that thought of two numbers and said one makes the other up when asked for it later. Add
`"preserve_thinking": true` to `strata-<model>.json` and restart, or send `"chat_template_kwargs":
{"preserve_thinking": true}` with a request, as with llama.cpp. The server also remembers the thinking of the
answers it wrote, so an app that sends an answer back without it (the chat page, Open WebUI) still gets it, as long as
the answer's text comes back unchanged.

### Can several people use one server at once?

Yes. By default the server answers one request at a time and the others wait their turn. With `"parallel": N` in
`strata-<model>.json` (or `--parallel N` when you run setup) it answers up to N together, with every model.
On a card that holds most of the model's experts that writes more tokens a second in all: on an RTX 5090 four
Qwen3.6 chats at once came at 428 tokens/s together, one alone at 312 to 333. On a small card, where
most experts run on the CPU, it gains less: on the 8 GB laptop three Qwen3.6 chats at once came at 105 tokens/s
together, one after the other at 79. How it works, and how to check it on your PC:
[BATCHING.md](BATCHING.md#the-strata-poly-models).

### Who maintains it?

Me, and it's MIT licensed. Strata's engine inside it comes from Strata's releases, and the added engine is all in
`src/poly`. Issues and PRs are welcome.

# Groot 19B-A3B 261023 (experimental)

[Bartowski's Groot GGUFs](https://huggingface.co/bartowski/GrootTeam_Groot-19B-A3B-261023-GGUF) use `qwen35moe`,
already implemented by `strata-poly`. Setup offers three quantizations (Q4_K_M, Q6_K and Q8_0) and their optional
BF16 image projector.
Q4_K_M ran on Windows 11, an RTX 5060 Laptop 8 GB and 32 GB of RAM: text on 2026-11-02 (it answered, streamed,
kept its reasoning separate and called a tool) and images on 2026-11-03; [the measured speeds](#measured-on-an-rtx-5060-laptop-8-gb)
are below. Q6_K, Q8_0 and other hardware have not been run. Matching tensor headers alone are not a correctness or
speed measurement.

## Choose it during setup

Run `START-HERE.bat` on Windows or `./setup.sh` on Linux and choose **Groot 19B-A3B 261023** from the model menu.
If another model is already installed, pass `--setup` to return to the setup questions.
Or go straight to this family:

```powershell
python setup.py --family groot
```

Like Tiel Coder and Qwen3.6, setup checks the hardware, offers the shared context menu (8K, 32K, 64K and 128K),
asks which quantization to download and whether to include the image encoder, verifies the selected files,
extracts the tokenizer and writes the run config and launcher.
Q4_K_M is the default; Q6_K and Q8_0 are for a PC with more RAM.
The default context is 32K, the same as the other `strata-poly` families; an explicit `--context` overrides it.
Setup downloads from Hugging Face by default. `HF_ENDPOINT` selects a mirror (for example
`https://hf-mirror.com`); `--source modelscope` is opt-in and checks the file against the same SHA-256.

For an unattended install with a chosen context and images:

```powershell
python setup.py --family groot --model Q4_K_M --context 32768 --vision yes --yes --no-start
python polystrata.py --model groot-q4_k_m
```

For text only, replace `--vision yes` with `--vision no`;
the projector is then neither downloaded nor loaded. An existing model folder can be supplied with
`--gguf-dir "C:\Models\Groot"`. Keep the published filenames in that folder.
Subsequent starts reuse the downloaded files and saved config.

## Quantization choices

Select a size interactively, or pass its name, for example `--model Q6_K` or `--model Q8_0`. Each size has its own
config and launcher, just like Tiel Coder. All three are single-file GGUFs with an embedded MTP block and the same
vision-projector pairing.

| Size | Download (GB) | RAM beside an 8 GiB card (GB, estimate) |
| --- | ---: | ---: |
| **Q4_K_M** (default) | 12.5 | 17 |
| Q6_K | 16.8 | 21 |
| Q8_0 | 20.2 | 24 |

These are Bartowski's mixed-precision layouts, not uniform bit widths. The MTP layer uses Q4_0 except in Q8_0,
where it uses Q8_0. All three files' headers were checked for the same geometry and MTP tensors.
The other published files are not offered. BF16, Q4_0 and Q4_1 contain routed-expert types outside the existing
native GPU cache path (type 30 for BF16, type 3 in the two older Q4 formats); the smaller i-quants and 3- and
5-bit K-quants were left out to keep the choice simple, and an imatrix is calibration data, not a model.

The default download is 12,484,339,584 bytes (12.48 GB decimal); the optional projector adds 902,822,208 bytes
(0.90 GB). Allow extra disk space for the engine, Python packages, tokenizer and installation files.
All downloads are pinned to revision `0025df0d03fb385344394f4e5afd73ce47da0680` and checked by SHA-256.

The engine keeps expert tensors in the model's RAM-backed file cache and caches as many as fit in VRAM.
Q4_K_M's directory contains 9.94 GB of routed experts and 2.54 GB of other weight tensors.
Each size's VRAM estimate is its non-expert tensor bytes in GiB plus a 2.5 GiB budget for state, KV rows and scratch,
rounded up to 0.1 GiB. This gives Q4_K_M 4.9 GiB beside its experts — a conservative budget over the measured run
below, where the card held 3.5 GiB of experts and peaked at 7.05 GiB VRAM (whole machine).
The measured Q4_K_M RAM peak was 21.8 GiB, whole machine; setup's 17 GB figure beside an 8 GiB card is a fit
estimate. Q6_K's and Q8_0's figures are still estimates; nothing has measured them.
The budget uses the 32K context, including the draft block's KV rows (22,528 bytes per token over 11 attention layers).
Windows, the display, other applications, image encoding and longer requests need headroom.
The expert cache adapts to available VRAM; total model size is not a full-GPU fit claim.

Existing configs from the earlier GrootTeam upload still start as before. Run setup again to switch a size to
Bartowski's files. Model IDs and config names are unchanged; Bartowski's tokenizer/profile packs use a separate
folder so an old extracted template is not silently reused. The old files are not deleted.

## Measured on an RTX 5060 Laptop 8 GB

`python polystrata.py benchmark --model groot-19b-a3b-261023-q4_k_m` on 2026-11-02, main at `c4e12110`, the
ready-made `strata-poly` 0.1.45 (CUDA 13.0), Q4_K_M, 32,768-token context, Windows 11, Core Ultra 9 275HX
(15 threads for the experts outside VRAM), 32 GB of RAM, driver 617.42. One run on one machine; the short prompt
pays the first-request warm-up.

| Prompt, tokens | Prefill, tokens/s | Decode, tokens/s | MTP kept | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 21 | 74.6 | 96.9 | 80.7% | 6.84 GiB | 21.66 GiB |
| 2,952 | 1,296.4 | 97.2 | 83.7% | 7.05 GiB | 21.80 GiB |
| 10,640 | 1,904.8 | 95.4 | 84.4% | 7.05 GiB | 21.51 GiB |

VRAM and RAM are the whole machine's use at its highest during the request, not the engine's alone. 1,994 of
5,248 routed experts stayed in VRAM (3.5 GiB); the rest ran on the CPU. The file loaded in 7.4 s.
The benchmark's 256-token answers reached the limit while still thinking, so the decode speeds are thinking
tokens. A separate request with room to finish answered "hello from groot", a streamed request kept content and
reasoning separate, and a tool request returned `get_weather({"city":"Paris"})` with `finish_reason: tool_calls`.
The raw run is in
[bench/results/2026-11-02-groot-q4_k_m-rtx5060-laptop](../../bench/results/2026-11-02-groot-q4_k_m-rtx5060-laptop/README.md).

A rerun on 2026-11-03 on the `add-groot-family` branch (`ca522caa`, the same tree as `8ab2be06`) reproduced that
table: decode was 93.8-101.5 tok/s cold and 83.9-101.3 after one warm-up request, and prefill 1,446-1,929 tok/s on
the two long prompts. VRAM peaked at 7.09-7.18 GiB and RAM at 18.3-21.3 GiB, whole machine. The 21-token row is the
unstable one: 6.4 tok/s before a warm-up against 122.7 after, because a 21-token read is mostly set-up work.
The raw runs are in
[bench/results/2026-11-03-groot-q4_k_m-rtx5060-laptop](../../bench/results/2026-11-03-groot-q4_k_m-rtx5060-laptop/README.md).

### Draft and batch parity

With `--no-gpu-experts --expert-swaps 0 --cpu-share 0 --tail-skip 0`, the same prompt ids with the MTP block and
with `--no-draft` gave the same 64 tokens on all three test prompts (11, 13 and 638 tokens; 35/45, 30/39 and 33/58
guesses kept with MTP on). `--batch 3 --check-batch 3` answered a 3,000-token prompt, the same prompt without its
first token and without its first two, alone and then together: all 32 tokens matched for all three sequences, both
with every expert on the CPU and with setup's profile (1,869 of 5,120 experts in VRAM).
Raw rows: [bench/results/2026-11-03-groot-draft-parity](../../bench/results/2026-11-03-groot-draft-parity/README.md).

### Compared with llama.cpp

llama.cpp 0.4.1-dev, built from source at the pinned commit with CUDA on arch 120, read the same GGUF. On the same
4,000-token prompt and sampler settings, Strata with MTP read it at 1,881 tok/s against 625 for llama.cpp's best
arm (`--fit on --fit-target 512`) and decoded at 80.7 against 50.0 tok/s at temperature 0; at 0.7 it was 1,860
against 618 and 86.0 against 51.0. Time to first token was 2.15 s against 6.49 s. llama.cpp's own MTP arm
(`--spec-type draft-mtp`) decoded at 52.3 and 49.0 tok/s with 0.41-0.47 of its guesses kept, against Strata's
0.70-0.76. One round, two reps, 4K only; llama.cpp was not tuned beyond `--fit`.
Greedy 200-token answers: the code answer was identical character for character (813 characters); the story
diverged after 165 characters, and both stayed coherent.
Raw rows: [bench/results/2026-11-03-groot-q4_k_m-llamacpp-rtx5060-laptop](../../bench/results/2026-11-03-groot-q4_k_m-llamacpp-rtx5060-laptop/README.md).

## Hardware and shared engine features

Groot uses the same `strata-poly` path as Tiel Coder and Qwen3.6, not an RTX 5060-specific engine.
The existing NVIDIA CUDA and AMD HIP build/installation paths, hardware checks, adaptive expert caching,
CPU experts, conversation caching, parallel requests and multi-GPU layer splitting apply.
Backend limitations are unchanged: the image encoder needs the existing CUDA build or CPU placement
where offered, and this does not add a Qwen35 engine to Intel Arc/SYCL.

Use the usual options, for example `--parallel 2`, `--gpus all`, `--vision cpu` and `--context 65536`.
Setup's shared memory estimates decide how much of the experts the GPU can hold.
Larger cards can cache more experts; smaller cards rely more on RAM and CPU work.
The table above is one card and one size; longer contexts than its 10,640-token prompt are still unmeasured.

On Windows 11 with 32 GB RAM and an RTX 5060 8 GB, `--vision yes` automatically places image encoding on the CPU.
Choose a shorter context such as `--context 8192` if needed; `--context 4096` is also available explicitly for
an initial smoke test. These are choices, not a special limit or default for this model.
Image tokens, conversation history and generated output all share the context budget.
Do not treat Windows shared GPU memory or persistent SSD paging as a successful memory fit.

## Chat, tools and images

Setup extracts the tokenizer and embedded chat template from the model itself, not from another Qwen model.
The tokenizer declares `gpt2` with the `qwen35` pretokenizer. The image projector declares
`qwen3vl_merger` and projects to the text model's 2,048-wide embeddings.
Images use the existing server/web upload and OpenAI-compatible image-content path.
CPU image encoding is the automatic choice for an 8 GB card; `--vision gpu` explicitly requests GPU encoding.
Six fresh 768×512 test images on 2026-11-03 (a code and a random number of shapes) got 5 of 6 answers exactly
right: the codes 6/6 and the square counts 6/6, with one circle count wrong. A picture became 441 prompt tokens;
the first request took 5.5 s because it starts the CPU encoder, the next five 4.2-4.7 s, and VRAM stayed at the
text-only 6.9-7.1 GiB. Raw rows and the images:
[bench/results/2026-11-03-groot-vision-rtx5060-laptop](../../bench/results/2026-11-03-groot-vision-rtx5060-laptop/README.md).

Once running, use the local web app at `http://127.0.0.1:8080`, the OpenAI API under `/v1`, or Anthropic
messages at `/v1/messages`. The API model ID is `groot-19b-a3b-261023-q4_k_m`.
Keep the server bound to `127.0.0.1`; use `--api-key` before exposing it to other machines.

The model card describes thinking by default, API/template-controlled non-thinking mode, and Qwen-style tool calls.
It does not support `/think` or `/nothink` conversation switches. Verify reasoning separation, streamed tool
arguments and tool-result continuation with your client before relying on them.
The card recommends different sampling settings for general thinking, coding and non-thinking requests;
consult it rather than changing PolyStrata's global defaults.

The GGUF contains one MTP draft block. Like Tiel Coder and Qwen3.6, setup leaves detection and speculative decoding
to the shared engine rather than forcing `--no-draft`. To compare ordinary decoding, add `--no-draft` to the
generated model config and restart. The registry reports MTP as disabled while that flag is present.
Groot does not set an approximate tail-skipping override.

## What remains to be measured

- Compare text logits and tokenizer IDs against a compatible reference runtime, and chunked prefill at more than
  one prompt size; greedy answers (identical for the 200-token code answer) and batch parity were compared, not
  logits.
- Exercise the Anthropic API, multi-turn tools, model switching and explicit context-limit errors; streaming, one
  tool call and the OpenAI API were checked.
- Test several pictures in one request, GPU image encoding, images with conversation history, and image answers on
  a second card; one picture at a time on the CPU encoder was checked.
- Measure Q6_K and Q8_0, other cards, longer contexts and repeated requests.

The Q4_K_M table above is the first measured RTX 5060 throughput and RAM/VRAM peak. The model's trained context is 262,144 tokens;
its card recommends at least 128K for complex reasoning. A 4K/8K smoke test does not establish that capability
or reproduce the card's benchmark results.

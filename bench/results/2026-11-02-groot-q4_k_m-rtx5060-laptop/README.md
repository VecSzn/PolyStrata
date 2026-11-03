# Groot 19B-A3B 261023 Q4_K_M on an RTX 5060 Laptop 8 GB

First measured run of Groot on `strata-poly`, 2026-11-02. One machine and one run; these are measurements, not
estimates.

- Model: `GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf`, 12,484,339,584 bytes, SHA-256
  `c941e12a231b40b8fb9f6ad7f6601878d72db39c2c62aefb185fbe5c7d0c2e29` (the pinned Hugging Face revision).
  Setup's default is Hugging Face; `huggingface.co` was unreachable from this network, so the file came from
  ModelScope, whose published SHA-256 is the same.
- Engine: the ready-made `strata-poly` 0.1.45 (`BUILD.json`, CUDA 13.0); repository main at `c4e12110`.
- PC: RTX 5060 Laptop 8 GB, driver 617.42; Core Ultra 9 275HX, 15 threads for the experts outside VRAM;
  32 GB of RAM; Windows 11 Home 10.0.26300; a system-managed 2.9 GB page file.
- Settings: context 32,768, Q4_K_M, MTP on, conversation cache `auto`, temperature 0.
- Command: `python polystrata.py benchmark --model groot-19b-a3b-261023-q4_k_m`
  (the default prompts of 21, 2,952 and 10,640 tokens, 256-token answers).

| Prompt, tokens | Prefill, tokens/s | Decode, tokens/s | MTP kept | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 21 | 74.6 | 96.9 | 80.7% | 6.84 GiB | 21.66 GiB |
| 2,952 | 1,296.4 | 97.2 | 83.7% | 7.05 GiB | 21.80 GiB |
| 10,640 | 1,904.8 | 95.4 | 84.4% | 7.05 GiB | 21.51 GiB |

VRAM and RAM are the whole machine's use at its highest while the request ran, not the engine's alone. The
21-token row pays the first-request warm-up. The 256-token answers reached the limit while still thinking, so
the decode speeds are thinking tokens, not answers. The file loaded in 7.4 s; 1,994 of 5,248 routed experts
stayed in VRAM (3.5 GiB) and the rest ran on the CPU.

Checked separately: a request with room to finish answered "hello from groot", a streamed request kept content
and reasoning separate, and a tool request returned `get_weather({"city":"Paris"})` with
`finish_reason: tool_calls`.

Not checked: images (the projector was not downloaded), Q6_K, Q8_0, other cards, contexts longer than the
10,640-token prompt, repeated requests, and a logits comparison with another runtime.

The raw benchmark output is in `benchmark.json`.

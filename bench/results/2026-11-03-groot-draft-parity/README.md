# Groot Q4_K_M draft and batch parity (2026-11-03)

Does Groot's embedded MTP block change what the model says, and does answering three sequences together change it?
Greedy decoding on one RTX 5060 Laptop 8 GB, `strata-poly` 0.1.45, Q4_K_M, 32,768-token context, 15 threads.

## Draft on / off

Same token ids through the engine twice, once with the MTP block (the default) and once with `--no-draft`, 64 new
tokens each. `--no-gpu-experts --expert-swaps 0 --cpu-share 0 --tail-skip 0 --window 8` keeps every routed expert on
the CPU in both arms, so the expert cache cannot differ between them; `--window 8` still lets the block guess.

| Prompt | Prompt tokens | Same 64 tokens | MTP guesses kept (on) |
| --- | ---: | :---: | ---: |
| "Explain in three sentences why the sky is blue." | 11 | yes | 35 of 45 |
| "Write a Python function that merges overlapping intervals, then explain it." | 13 | yes | 30 of 39 |
| README excerpt + "what does this software do?" | 638 | yes | 33 of 58 |

All three 64-token answers were identical token for token. The guesses change speed, not the text.

## Three sequences together

`--batch 3 --check-batch 3 --tokens <3,000 ids> --max-new 32`, with `--expert-swaps 0 --cpu-share 0 --window 1
--tail-skip 0` so the expert cache and guesses are out of the comparison. `--check-batch` answers the prompt on its
own, then the same prompt without its first token and without its first two, one after the other and then together,
and compares them token by token.

| Expert placement | Experts in VRAM | Same 32 tokens, each sequence | 3 prompts read | 32 steps of 3 |
| --- | ---: | :---: | ---: | ---: |
| `--no-gpu-experts` | 0 of 5,120 | 32/32, 32/32, 32/32 | 80.4 s | 1.40 s (68.6 tok/s) |
| Setup's profile | 1,869 of 5,120 (3.2 GiB) | 32/32, 32/32, 32/32 | 5.0 s | 1.03 s (93.1 tok/s) |

Every token matched in both configurations, including with the GPU expert cache in use. The load took 3.8 s with all
experts on the CPU and 5.5 s with the profile.

Raw rows: `draft-parity.json`, `batch-check-8k-cpu-experts.log`, `batch-check-8k-gpu-experts.log`. Not checked:
temperature above 0, prompts longer than 3,000 tokens, more than three sequences.

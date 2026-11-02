# Clover-35B-A3B-Preview

Back to [README.md](../../README.md)

**Clover-35B-A3B-Preview** by DiffuSpace, a file of 13 to 21 GB. It is Qwen3.6-35B-A3B trained further to also
write a block of tokens at once. It reads text only.

## How fast is it?

Measured for PolyStrata on the laptop, with a short chat and a prompt of 32,000 tokens. Its card has 8 GB, and the
processor runs the experts the card has no room for.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q3_K_M** | 68-73 tokens/s | 1,560-1,700 tokens/s |
| **APEX-I-NANOPLUS** | 72-76 tokens/s | 1,530-1,690 tokens/s |
| **Q4_K_M** | 38-58 tokens/s | 1,220-1,360 tokens/s |

</td></tr>
</table>

After the prompt of 32,000 tokens the laptop writes at 60 to 65 tokens/s with Q3_K_M. Q4_K_M fills the 32 GB of RAM
there, which is what makes it slow and uneven. The workstation with the RTX 5090 was not measured.

Qwen3.6-35B-A3B writes faster on the same laptop, 94 to 100 tokens/s, because its file has a draft block that guesses
the next words for one layer's work. Clover's files have none. Clover can guess a block of words itself, but that runs
the whole model once more, and PolyStrata leaves it off: on a card that held every expert the answers came sooner
without it ([measured](../BACKENDS.md#clover-35b-a3b-preview)).

On an RTX PRO 6000 that held every expert of Q3_K_M, PolyStrata picked the same next token as llama.cpp (the
`clover_moe` support of its pull request 31306) at 127 of 128 steps after a short prompt and at 125 of 128 after
5,573 tokens.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 3.5 GB are taken before any expert. It has run on a laptop's 8 GB card and on a 96 GB card. It has not been tried on an AMD card. |
| **RAM** | About **16, 20 or 25 GB** beside an 8 GB card for APEX-I-NANOPLUS, Q3_K_M and Q4_K_M, the installer's estimate. A card with 24 GB holds the two smaller sizes whole. All three have run on a laptop with 32 GB. |
| **Disk** | 13, 17 or 21 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as for [Qwen3.6-35B-A3B](qwen36.md). maternion's files carry no chat template,
so the installer downloads the model's own from DiffuSpace. Clover's license is Apache 2.0.

## Which model should I pick?

Three sizes, all made with the converter of llama.cpp's pull request 31306. The installer takes Q3_K_M, or
APEX-I-NANOPLUS when the PC has too little RAM for it. Q4_K_M is chosen by name: `--model Q4_K_M`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **Q3_K_M** | the whole model with 3-bit experts ([maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)) | 17 GB | about 20 GB |
| **APEX-I-NANOPLUS** | 2- and 3-bit experts from an importance matrix ([IsValorum](https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF)): the least RAM | 13 GB | about 16 GB |
| **Q4_K_M** | 4-bit experts (maternion) | 21 GB | about 25 GB |

maternion's two sizes were quantized without an importance matrix. How close each size comes to the model as trained
was not measured.

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#clover-35b-a3b-preview).

# Gemma 4 26B-A4B

Back to [README.md](../../README.md)

**Gemma 4 26B-A4B**, a file of 11 to 17 GB. It also reads pictures.

## How fast is it?

Measured the same way as [Qwen3.6-35B-A3B](qwen36.md), on the same two PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 92-93 tokens/s | 2,240 tokens/s |
| **UD-Q2_K_XL** | 101-104 tokens/s | 2,150-2,160 tokens/s |
| **UD-Q4_K_M** | 85 tokens/s | 2,150 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 353-356 tokens/s | 8,420-8,480 tokens/s |
| **UD-Q2_K_XL** | 332-338 tokens/s | 7,940-7,970 tokens/s |
| **UD-Q4_K_M** | 335-346 tokens/s | 8,320-8,390 tokens/s |
| **QAT-UD-Q4_K_XL** | 356-371 tokens/s | 8,460-8,680 tokens/s |

</td></tr>
</table>

A prompt of 10,000 tokens is read at 2,130 tokens/s on the laptop and at 6,800 on the workstation. QAT-UD-Q4_K_XL was
measured later, with the engine of v0.1.45-poly.4; UD-Q4_K_M measured beside it with that engine wrote 333-353 tokens/s
and read 8,510-8,570. On the laptop, with other programs open and RAM short, it was run in turn with UD-Q4_K_M: it wrote 50-62 tokens/s in a chat against 47-48, read the long prompt at 1,680-1,730 against 1,660-1,670, and wrote 65-68 after it against 48-49.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 6 GB are taken before any expert. It has run on a 32 GB card, on that card held to 8 GB, and on a laptop's 8 GB card. On an AMD card it can run as a beta on Linux and under Windows: setup compiles the engine on your PC; users have run Gemma 4 on an RX 6700 XT (12 GB) under Linux. |
| **RAM** | About **15, 18, 19 or 21 GB** beside an 8 GB card for the four sizes, the installer's estimate. By the same estimate a card with 16 GB holds the two smaller sizes and the QAT file whole. All four sizes have run on a laptop with 32 GB. |
| **Disk** | 11, 13, 15 or 17 GB for the model file and its draft model, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it downloads the engine ready-made
from the latest release. On a processor without AVX2, or in a copy whose engine changed after that release, it
compiles the engine instead with Visual Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not
yet: two cards on Windows, three cards or more. On an AMD card under Windows setup compiles `strata-poly` for the
card as it does on Linux (AMD's TheRock wheels in `.rocm-win`) and warns that runs there are new
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Which model should I pick?

Four sizes so far. The installer takes UD-Q3_K_M, or UD-Q2_K_XL when the PC has too little RAM for it. The other two are chosen by name: `--model UD-Q4_K_M`, `--model QAT-UD-Q4_K_XL`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | the whole model with 3-bit experts ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)), and Google's draft model for it | 13 GB | about 18 GB |
| **UD-Q2_K_XL** | the same with 2-bit experts: the least RAM | 11 GB | about 15 GB |
| **UD-Q4_K_M** | the same with 4- and 5-bit experts | 17 GB | about 21 GB |
| **QAT-UD-Q4_K_XL** | Google's quantization-aware trained weights at 4 bits ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)), with their own draft model | 15 GB | about 19 GB |

A larger card holds more of the model and needs less RAM: with UD-Q3_K_M about 14 GB beside a 12 GB card, and 10 GB once the card
holds all of it. These are the installer's estimates. They have not been checked on such a PC.

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#gemma-4-26b-a4b).

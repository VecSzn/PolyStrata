# Ornith 1.5

Back to [README.md](../../README.md)

**Ornith 1.5**, which has Qwen3.6-35B-A3B's architecture and other weights: a file of 14 to 24 GB. It also reads pictures.

## How fast is it?

Measured the same way as [Qwen3.6-35B-A3B](qwen36.md), on the same two PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 76-82 tokens/s | 1,620 tokens/s |
| **APEX-I-MINI** | 80-86 tokens/s | 1,550-1,560 tokens/s |
| **APEX-I-QUALITY** | 59-63 tokens/s | 1,570-1,580 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 316-319 tokens/s | 8,040-8,110 tokens/s |
| **APEX-I-MINI** | 289-300 tokens/s | 7,710-7,740 tokens/s |
| **APEX-I-QUALITY** | 297-302 tokens/s | 7,860-7,880 tokens/s |

</td></tr>
</table>

A prompt of 10,000 tokens is read at 1,590 tokens/s on the laptop and at 6,400 on the workstation.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 4 GB are taken before any expert. It has run on a 32 GB card, on that card held to 8 GB, and on a laptop's 8 GB card. On an AMD card it can run as a beta on Linux and under Windows: setup compiles the engine on your PC; tested so far with Qwen3.6 and Ornith on an RX 7600. |
| **RAM** | About **18, 22 or 28 GB** beside an 8 GB card for the three sizes, the installer's estimate. A card with 24 GB holds the two smaller sizes whole. All three sizes have run on a laptop with 32 GB. |
| **Disk** | 14, 17 or 24 GB for the model file, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it downloads the engine ready-made
from the latest release. On a processor without AVX2, or in a copy whose engine changed after that release, it
compiles the engine instead with Visual Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not
yet: two cards on Windows, three cards or more. On an AMD card under Windows setup compiles `strata-poly` for the
card as it does on Linux (AMD's TheRock wheels in `.rocm-win`) and warns that runs there are new
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Which model should I pick?

Three sizes so far. The installer takes APEX-I-COMPACT, or APEX-I-MINI when the PC has too little RAM for it. APEX-I-QUALITY is chosen by name: `--model APEX-I-QUALITY`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | the whole model with 3- and 4-bit experts ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) by mudler) | 17 GB | about 22 GB |
| **APEX-I-MINI** | the same with 2- and 3-bit experts: the least RAM | 14 GB | about 18 GB |
| **APEX-I-QUALITY** | the same with 4- to 6-bit experts | 24 GB | about 28 GB |

A larger card holds more of the model and needs less RAM: with APEX-I-COMPACT about 17 GB beside a 12 GB card, 13 GB beside a 16 GB
card, and 10 GB once the card holds all of it. These are the installer's estimates. They have not been checked on
such a PC.

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#ornith-15).

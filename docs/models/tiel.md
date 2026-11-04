# Tiel Coder 35B-A3B

Back to [README.md](../../README.md)

**Tiel Coder 35B-A3B**, Ornith 1.5 quantized for coding agents by peculiar-ragdoll: a file of 14 to 23 GB. It also reads pictures.

## How fast is it?

Measured the same way as [Qwen3.6-35B-A3B](qwen36.md), on the same two PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 83 tokens/s | 1,700 tokens/s |
| **UD-IQ3_XXS** | 96-97 tokens/s | 1,610 tokens/s |
| **UD-Q4_K_XL** | 74-79 tokens/s | 1,660-1,670 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 337-345 tokens/s | 8,080-8,110 tokens/s |
| **UD-IQ3_XXS** | 352 tokens/s | 7,520 tokens/s |
| **UD-Q4_K_XL** | 316-322 tokens/s | 7,750-7,970 tokens/s |

</td></tr>
</table>

A prompt of 10,000 tokens is read at 1,670 tokens/s on the laptop and at 6,300 on the workstation.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 5 GB are taken before any expert. It has run on a 32 GB card, on that card held to 8 GB, and on a laptop's 8 GB card. On an AMD card it can run as a beta on Linux and under Windows: setup compiles the engine on your PC; users have run Tiel Coder on an RX 6700 XT and an RX 7800 XT under Linux, and on an RX 9070 XT under Windows. |
| **RAM** | About **18, 21 or 27 GB** beside an 8 GB card for the three sizes, the installer's estimate. A card with 24 GB holds all three sizes whole. All three sizes have run on a laptop with 32 GB. |
| **Disk** | 14, 17 or 23 GB for the model file, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it downloads the engine ready-made
from the latest release. On a processor without AVX2, or in a copy whose engine changed after that release, it
compiles the engine instead with Visual Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not
yet: two cards on Windows, three cards or more. On an AMD card under Windows setup compiles `strata-poly` for the
card as it does on Linux (AMD's TheRock wheels in `.rocm-win`) and warns that runs there are new
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Which model should I pick?

Three sizes so far. The installer takes UD-Q3_K_XL, or UD-IQ3_XXS when the PC has too little RAM for it. UD-Q4_K_XL is chosen by name: `--model UD-Q4_K_XL`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q3_K_XL** | the whole model with 3- and 4-bit experts ([peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)) | 17 GB | about 21 GB |
| **UD-IQ3_XXS** | the same with 2- and 3-bit experts: the least RAM | 14 GB | about 18 GB |
| **UD-Q4_K_XL** | the same with 4- to 6-bit experts | 23 GB | about 27 GB |

A larger card holds more of the model and needs less RAM: with UD-Q3_K_XL about 17 GB beside a 12 GB card, 13 GB beside a 16 GB
card, and 10 GB once the card holds all of it. These are the installer's estimates. They have not been checked on
such a PC.

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#tiel-coder-35b-a3b).

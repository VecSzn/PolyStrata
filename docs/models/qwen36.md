# Qwen3.6-35B-A3B

Back to [README.md](../../README.md)

**Qwen3.6-35B-A3B**, a file of 13 to 23 GB. It also reads pictures.

## How fast is it?

Measured for PolyStrata on two PCs, with a short chat and a prompt of 30,000 to 32,000 tokens: on the left a laptop
whose card has 8 GB, where the processor runs the experts the card has no room for, and on the right a workstation
whose card holds the whole model.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 94-100 tokens/s | 1,690 tokens/s |
| **UD-Q2_K_XL** | 123-124 tokens/s | 1,590 tokens/s |
| **UD-Q4_K_M** | 82-86 tokens/s | 1,560-1,570 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 394-395 tokens/s | 7,800-7,810 tokens/s |
| **UD-Q2_K_XL** | 450 tokens/s | 7,720-7,740 tokens/s |
| **UD-Q4_K_M** | 395-396 tokens/s | 8,070-8,080 tokens/s |

</td></tr>
</table>

A prompt of 10,000 tokens is read at 1,660 tokens/s on the laptop and at 6,300 on the workstation.
The workstation's card held to 8 GB, a stand-in from before the laptop was measured,
is in [docs/BACKENDS.md](../BACKENDS.md#with-8-gb-of-vram).

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 5 GB are taken before any expert. It has run on a 32 GB card, on that card held to 8 GB, and on a laptop's 8 GB card. On an AMD card it can run as a beta on Linux and under Windows: setup compiles the engine on your PC; tested so far with Qwen3.6 and Ornith on an RX 7600. |
| **RAM** | About **17, 21 or 27 GB** beside an 8 GB card for the three sizes, the installer's estimate. A card with 24 GB holds the two smaller sizes whole. All three sizes have run on a laptop with 32 GB. |
| **Disk** | 13, 17 or 23 GB for the model file, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it downloads the engine ready-made
from the latest release. On a processor without AVX2, or in a copy whose engine changed after that release, it
compiles the engine instead with Visual Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not
yet: two cards on Windows, three cards or more. On an AMD card under Windows setup compiles `strata-poly` for the
card as it does on Linux (AMD's TheRock wheels in `.rocm-win`) and warns that runs there are new
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

Since its 0.1.44 the [Strata](https://github.com/Niko1221/Strata) project's own engine runs Qwen3.6 too, from UD-Q4_K_XL up
(`--family qwen36-strata`, 32 GB of RAM or more). On an RTX 5090 with UD-Q4_K_XL it read a 4K prompt 1.6 times as fast
as strata-poly; a 31K prompt and writing came out about even ([measured](../BACKENDS.md#qwen36-on-two-engines)). The smaller sizes, for
8 GB cards, stay on strata-poly.

## Which model should I pick?

Three sizes so far. The installer takes UD-Q3_K_M, or UD-Q2_K_XL when the PC has too little RAM for it. UD-Q4_K_M is chosen by name: `--model UD-Q4_K_M`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | the whole model with 3-bit experts ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 GB | about 21 GB |
| **UD-Q2_K_XL** | the same with 2- and 3-bit experts: the least RAM | 13 GB | about 17 GB |
| **UD-Q4_K_M** | the same with 4-bit experts | 23 GB | about 27 GB |

A larger card holds more of the model and needs less RAM: with UD-Q3_K_M about 17 GB beside a 12 GB card, 13 GB beside a 16 GB
card, and 10 GB once the card holds all of it. These are the installer's estimates. They have not been checked on
such a PC.

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#qwen36-35b-a3b).

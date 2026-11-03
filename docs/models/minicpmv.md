# MiniCPM-V 4.7 35B-A3B

Back to [README.md](../../README.md)

**MiniCPM-V 4.7 35B-A3B** by OpenBMB, which has Qwen3.6-35B-A3B's architecture and weights of its own, made to read pictures: a file of 16 to 29 GB.

## How fast is it?

Measured the same way as [Qwen3.6-35B-A3B](qwen36.md), on the same two PCs. Its file has no draft block, so it writes one token at a time.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 64 tokens/s | 1,150 tokens/s |
| **mix_q3gud_im** | 59 tokens/s | 1,130-1,150 tokens/s |
| **Q6_K** | 40 tokens/s | 960-980 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 308-311 tokens/s | 7,910-7,950 tokens/s |
| **mix_q3gud_im** | 291-293 tokens/s | 7,730-7,740 tokens/s |
| **Q6_K** | 277-278 tokens/s | 7,290-7,330 tokens/s |

</td></tr>
</table>

A prompt of 10,000 tokens is read at 1,260 tokens/s on the laptop and at 8,790 on the workstation. On the laptop Q6_K
fills the 32 GB of RAM, which is what makes it slow there.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 3.5 GB are taken before any expert. It has run on a 32 GB card and on a laptop's 8 GB card. On an AMD card it can run as a beta on Linux and under Windows: setup compiles the engine on your PC; it has run on an RX 7600. |
| **RAM** | About **19, 25 or 31 GB** beside an 8 GB card for the three sizes, the installer's estimate. A card with 32 GB holds all three sizes whole. All three sizes have run on a laptop with 32 GB. |
| **Disk** | 16, 21 or 29 GB for the model file, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it downloads the engine ready-made
from the latest release. On a processor without AVX2, or in a copy whose engine changed after that release, it
compiles the engine instead with Visual Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not
yet: two cards on Windows, three cards or more. On an AMD card under Windows setup compiles `strata-poly` for the
card as it does on Linux (AMD's TheRock wheels in `.rocm-win`) and warns that runs there are new
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Which model should I pick?

Three sizes so far, from OpenBMB. The installer takes Q4_K_M, or mix_q3gud_im when the PC has too little RAM for it.
Q6_K is chosen by name: `--model Q6_K`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **Q4_K_M** | the whole model with 4-bit experts ([OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)) | 21 GB | about 25 GB |
| **mix_q3gud_im** | the same with 3- and 4-bit experts: the least RAM | 16 GB | about 19 GB |
| **Q6_K** | the same with 6-bit experts | 29 GB | about 31 GB |

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#minicpm-v-47-35b-a3b).

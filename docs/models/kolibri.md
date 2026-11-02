# Kolibri 1

Back to [README.md](../../README.md)

**Kolibri 1** by Aleph Alpha, a model of 19,200 experts that writes German as well as English: a file of 29 to 48 GB,
most of which stays in RAM beside a small card. Four layers in five look back only 513 tokens.

## How fast is it?

Measured the same way as [Flint 40B-A2.5B](flint.md), with a prompt of 31,676 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q4_K_M** | Not measured | Not measured |
| **Q2_K** | Not measured | Not measured |
| **Q3_K_M** | Not measured | Not measured |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_K** | 189-192 tokens/s | 4,980-5,070 tokens/s |

</td></tr>
</table>

The workstation's card holds all 19,200 experts of Q2_K (25.6 GiB). It started answering the 31,676-token prompt after
6.3 to 6.4 seconds and wrote after it at 129-131 tokens/s. Q4_K_M and Q3_K_M were not run there, and no size was run on the
laptop.

The pull request that added the model measured it on other PCs. On an RTX 3060 (12 GB) with 24 GB of RAM under
Windows, Q2_K wrote a short chat at 21 tokens/s and read a 32K prompt at 698. On one Tesla V100 (32 GB) Q4_K_M wrote
at 91 tokens/s and read a 32K prompt at 1,017; on two it read at 2,179.

Against llama.cpp with the conversion's `kolibri1` patch, following its greedy answer to an English and a German
prompt, Q2_K picks the same most likely token at 120 and 126 of 128 steps. Where the two differ, llama.cpp's token is
this engine's second or third choice at all but one step, a near tie.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 4 GB are taken before any expert. It has run on a 12 GB card, a 32 GB Tesla V100 and a 32 GB RTX 5090. |
| **RAM** | About **52, 33 or 42 GB** beside an 8 GB card for Q4_K_M, Q2_K and Q3_K_M, the installer's estimate. Beside a 32 GB card Q2_K needs about 10 GB and Q4_K_M about 26 GB. |
| **Disk** | 29 to 48 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as for [Qwen3.6-35B-A3B](qwen36.md). It has not been tried on an AMD card. It
reads text only, and it has no draft model, so it writes one token at a time. Its license is Apache 2.0.

## Which model should I pick?

Three sizes, from [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF)'s conversion. Beside an 8 GB card the
installer takes Q4_K_M from about 48 GB of RAM and Q2_K below that; beside a larger card both need less. Q3_K_M is
chosen by name: `--model Q3_K_M`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **Q4_K_M** | 4-bit, with 6-bit down projections in half the layers | 48 GB | about 52 GB |
| **Q2_K** | 2-bit: the least RAM | 29 GB | about 33 GB |
| **Q3_K_M** | 3-bit | 38 GB | about 42 GB |

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#kolibri-1).

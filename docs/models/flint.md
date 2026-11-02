# Flint 40B-A2.5B

Back to [README.md](../../README.md)

**Flint 40B-A2.5B** by Recursive, made for coding agents: a file of 18 to 42 GB. Most of its layers look back only 513 tokens, so a long conversation takes little VRAM: a 128K context takes 2.6 GB more than 32K.

## How fast is it?

Measured the same way as [LFM2.5-24B-A2B](lfm25.md), on the same two PCs, with a prompt of about 29,000 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 90 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 20-29 tokens/s | 1,310-1,420 tokens/s |
| **UD-Q3_K_XL** | 61-63 tokens/s | 2,000-2,010 tokens/s |
| **UD-Q6_K_XL** | Does Not Fit | Does Not Fit |
| **Q8_0** | Does Not Fit | Does Not Fit |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 251-252 tokens/s | 12,900 tokens/s |
| **UD-Q3_K_XL** | 260-264 tokens/s | 13,080-13,200 tokens/s |
| **Q8_0** | 158-160 tokens/s | 6,740-6,970 tokens/s |

</td></tr>
</table>

On the laptop only UD-Q3_K_XL stays in the 32 GB of RAM beside the desktop's programs. The larger sizes keep reading
experts from the SSD, which is what makes them slow there, and the installer takes UD-Q3_K_XL there. The
workstation's card holds UD-Q4_K_XL and UD-Q3_K_XL whole; UD-Q6_K_XL was not run there.

On the laptop llama.cpp with `--fit on`, the fastest of the settings tried there, read a prompt of 29,442 tokens at
390 tokens/s and wrote after it at 35; PolyStrata read it at 1,950 to 2,010 and wrote at 49 to 52
([the runs](../../bench/results/2026-10-24-laptop-flint-llamacpp-a37cc46/README.md)).

On the workstation llama-bench of llama.cpp read 29,816 tokens at 7,790 tokens/s with its default batch of 512 and
at 12,470 with 2,048, where PolyStrata reads at 12,910-12,940; one token at a time llama.cpp wrote at 222 tokens/s and at
188 after that prompt, PolyStrata at 249-256 and 212-213. A prompt of 124,122 tokens in a 128K context was read and
answered in 20 seconds, and the three facts hidden in it were found.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 4 GB are taken before any expert. It has run on a laptop's 8 GB card and on a 32 GB card. |
| **RAM** | About **22, 29, 40 or 46 GB** beside an 8 GB card for UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL and Q8_0, the installer's estimate. On a laptop with 32 GB only UD-Q3_K_XL stayed in RAM, so the installer takes UD-Q4_K_XL from about 40 GB on. A 32 GB card holds UD-Q4_K_XL whole. |
| **Disk** | 18 to 42 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as for [Qwen3.6-35B-A3B](qwen36.md). It has not been tried on an AMD card. It reads
text only. Its license is Apache 2.0.

The full list: [docs/FLINT.md](../FLINT.md#what-you-need).

## Which model should I pick?

Four sizes. Beside an 8 GB card the installer takes UD-Q4_K_XL from about 40 GB of RAM and UD-Q3_K_XL below that;
beside a larger card UD-Q4_K_XL needs less. The others are chosen by name: `--model Q8_0`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | the whole model with 4- and 5-bit experts ([Unsloth](https://huggingface.co/unsloth/flint-GGUF)) | 25 GB | about 29 GB |
| **UD-Q3_K_XL** | the same with 3-bit experts: the least RAM | 18 GB | about 22 GB |
| **UD-Q6_K_XL** | 6-bit experts | 36 GB | about 40 GB |
| **Q8_0** | 8-bit throughout, the closest to the model as trained | 42 GB | about 46 GB |

On the 32 GB laptop UD-Q4_K_XL's file did not stay in RAM: it wrote a short chat at 20-29 tokens/s, UD-Q3_K_XL at
61-63. Against Q8_0, on five prompts of 256 tokens each, the other sizes pick Q8_0's most likely token at 97.8% of
the steps (UD-Q6_K_XL), 95.9% (UD-Q4_K_XL) and 94.0% (UD-Q3_K_XL).

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#flint-40b-a25b).

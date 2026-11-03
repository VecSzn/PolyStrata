# LFM2.5-24B-A2B

Back to [README.md](../../README.md)

**LFM2.5-24B-A2B** by Liquid AI, a file of 11 to 25 GB. Most of its layers have no attention, so a long conversation takes little VRAM: a 128K context takes 2 GB more than 32K. Its license allows commercial use only to companies with less than 10 million dollars of yearly revenue.

## How fast is it?

Measured the same way as [Qwen3.6-35B-A3B](qwen36.md), on the same two PCs, with a prompt of about 30,000 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 70-73 tokens/s | 3,590-3,620 tokens/s |
| **UD-Q3_K_XL** | 88 tokens/s | 3,760-3,810 tokens/s |
| **QAD-Q4_0** | 77-78 tokens/s | 3,720-3,730 tokens/s |
| **UD-Q6_K_XL** | 42-43 tokens/s | 3,030-3,120 tokens/s |
| **Q8_0** | 23-34 tokens/s | 2,950-3,040 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 480-485 tokens/s | 23,390-23,460 tokens/s |
| **UD-Q3_K_XL** | 490-497 tokens/s | 22,920-23,400 tokens/s |
| **QAD-Q4_0** | 498 tokens/s | 23,880-23,900 tokens/s |
| **UD-Q6_K_XL** | 423-426 tokens/s | 21,250-21,270 tokens/s |
| **Q8_0** | 376 tokens/s | 23,270-23,350 tokens/s |

</td></tr>
</table>

On the laptop Q8_0 fills the 32 GB of RAM, which is what makes it slow there; the workstation holds every size
whole.

With `--n-cpu-moe 25`, the best of the settings tried on the laptop, llama.cpp b11540 read a prompt of 32,000 tokens
there at 530 tokens/s and wrote after it at 53; PolyStrata read it at 3,540 and wrote at 71 to 73. On the workstation
llama-bench read 29,816 tokens at 13,300 tokens/s with llama.cpp's default batch of 512 and at 23,600 with 2,048, as
fast as PolyStrata, and wrote one token at a time at 455 tokens/s and at 369 after that prompt, where PolyStrata
writes at 452 and 387. A prompt of about 124,000 tokens in a 128K context was read and answered in 8.7 seconds on the
workstation; the laptop read it in 45 seconds.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 4 GB are taken before any expert. It has run on a laptop's 8 GB card and on a 32 GB card. |
| **RAM** | About **16, 18, 20, 27 or 31 GB** beside an 8 GB card for UD-Q3_K_XL, QAD-Q4_0, UD-Q4_K_XL, UD-Q6_K_XL and Q8_0, the installer's estimate. A card with 24 GB holds the three smaller sizes whole. All five have run on a laptop with 32 GB. |
| **Disk** | 11 to 25 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as for [Qwen3.6-35B-A3B](qwen36.md). A user has run it on an AMD RX 6700 XT
(12 GB) under Linux. It reads text only. Liquid AI's license allows commercial use only to companies with less than 10 million dollars of yearly
revenue ([LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)).

The full list: [docs/LFM.md](../LFM.md#what-you-need).

## Which model should I pick?

Five sizes. The installer takes UD-Q4_K_XL, or UD-Q3_K_XL when the PC has too little RAM for it. The others are
chosen by name: `--model QAD-Q4_0`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | the whole model with 4-bit experts ([Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)) | 15 GB | about 20 GB |
| **UD-Q3_K_XL** | the same with 3-bit experts: the least RAM | 11 GB | about 16 GB |
| **QAD-Q4_0** | 4-bit, from [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF), trained to lose less to that quantization | 14 GB | about 18 GB |
| **UD-Q6_K_XL** | 6-bit experts | 22 GB | about 27 GB |
| **Q8_0** | 8-bit throughout, the closest to the model as trained | 25 GB | about 31 GB |

Against Q8_0, on five prompts of 256 tokens each, the other sizes pick Q8_0's most likely token at 96.8% of the steps
(UD-Q6_K_XL), 95.5% (UD-Q4_K_XL), 92.7% (QAD-Q4_0) and 91.6% (UD-Q3_K_XL).

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#lfm25-24b-a2b).

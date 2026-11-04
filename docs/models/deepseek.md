# DeepSeek V4 Flash

Back to [README.md](../../README.md)

**DeepSeek V4 Flash**, a file of 91 GB, most of which stays in RAM too. Its memory of a conversation is small: a 128K context takes 0.7 GB more VRAM than 32K.

## How fast is it?

Measured for PolyStrata on the same workstation as [GLM 5.3 Flash](glm.md), with a short chat and a 30K-token prompt.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-IQ2_XXS** | 56 tokens/s | 1,040 tokens/s |

</td></tr>
</table>

On the same PC llama.cpp wrote 25 to 28 tokens per second and read a prompt of 3,384 tokens at 96. A prompt of
116,473 tokens in a 128K context was read and answered in 174 seconds, one of 487,529 tokens in a 512K context in 32
minutes, and both times the three facts hidden in it were found.
What was measured and how: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-it-was-measured-on).

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer. It has run on **32 GB of VRAM** only. About 11.5 GB are taken before any expert, so a 12 GB card cannot start it. |
| **RAM** | Beside a 32 GB card, **72 GB or more**, the installer's estimate. A smaller card needs more RAM ([how much](#which-model-should-i-pick)). It has run with 120 GB. |
| **Disk** | 91 GB for the model's three files, on an SSD. |
| **System** | Linux, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. Not tried yet: AMD cards, two cards.
It reads text only. A user runs it on Windows with one 24 GB card and 128 GB of RAM ([report](https://github.com/VecSzn/PolyStrata/issues/21)).

The full list: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-you-need).

## Which model should I pick?

One size so far.

| Size | What it is | Download | RAM beside a 32 GB card |
| --- | --- | ---: | ---: |
| **UD-IQ2_XXS** | the whole model at about 2 bits ([Unsloth](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)) | 91 GB | 72 GB |

As with GLM 5.3 Flash, the experts the card does not hold stay in RAM: about 80 GB beside a 24 GB card and 89 GB
beside a 16 GB card, by the installer's estimate, and a 12 GB card cannot start it. Only the 32 GB card has been
tried.

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#deepseek-v4-flash).

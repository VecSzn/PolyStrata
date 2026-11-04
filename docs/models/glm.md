# GLM 5.3 Flash

Back to [README.md](../../README.md)

**GLM 5.3 Flash**, the largest: a file of 79 to 114 GB, most of which stays in RAM.

## How fast is it?

Measured for PolyStrata on one workstation, with a short chat and a 29K-token prompt.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **3.0BIT** | 59-66 tokens/s | 820 tokens/s |
| **UD-IQ2_XXS** | 50-52 tokens/s | 1,060-1,080 tokens/s |
| **REAP50-Q3_K_M** | 59-66 tokens/s | 1,300-1,310 tokens/s |

</td></tr>
</table>

GLM 5.3 Flash is a much larger model than the others, and about a fifth of its experts fit in this card's 32 GB, so it writes more slowly.
What was measured and how: [docs/GLM.md](../GLM.md#what-it-was-measured-on).

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer. It has run on **32 GB of VRAM** only. About 12 GB are taken before any expert, so a 12 GB card cannot start it. |
| **RAM** | Beside a 32 GB card, **96 GB or more** for the whole model, 82 GB for its 2-bit size, 60 GB for the size with half the experts. A smaller card needs more RAM ([how much](#which-model-should-i-pick)). It has run with 120 GB. |
| **Disk** | 114 GB, 102 GB or 79 GB for the model file, on an SSD. |
| **System** | Linux, and a current NVIDIA driver. |

The installer sets up everything else. It compiles the engine on your PC and offers to install what that needs
(g++ and the CUDA toolkit). Not yet: AMD cards, two cards on Windows, three cards or more, pictures.
A user runs it on Windows with one 24 GB card and 128 GB of RAM ([report](https://github.com/VecSzn/PolyStrata/issues/10)).

The full list: [docs/GLM.md](../GLM.md#what-you-need).

## Which model should I pick?

Three sizes so far. The installer takes the first one your RAM is enough for.

| Size | What it is | Download | RAM beside a 32 GB card |
| --- | --- | ---: | ---: |
| **3.0BIT** | the whole model, at about 3 bits ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 GB | 96 GB |
| **UD-IQ2_XXS** | the whole model, at about 2 bits ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): it reads a prompt faster and writes more slowly | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | half of the routed experts removed ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) by patrickbdevaney) | 79 GB | 60 GB |

Your graphics card decides how much RAM it needs, because the experts the card does not hold have to stay in RAM.

| Your graphics card | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | the PC it has run on (120 GB of RAM) |
| **24 GB** | about 104 GB | about 90 GB | about 68 GB | not run yet |
| **16 GB** | about 113 GB | about 99 GB | about 77 GB | not run yet |
| **12 GB or less** | | | | it does not start |

The RAM numbers are the installer's estimate from the one PC it has run on. With less RAM it still starts, but it
reads experts from the SSD while it answers and gets much slower. The installer says so and asks first.

Other quantizations of GLM 5.3 Flash are not supported yet: [what is missing](../GLM.md#what-is-missing).

## How does it work?

The diagram and how it works: [HOW_IT_WORKS_MODELS.md](../HOW_IT_WORKS_MODELS.md#glm-53-flash).

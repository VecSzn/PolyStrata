# GLM 5.3 Flash

返回 [README.zh-CN.md](../../README.zh-CN.md)

**GLM 5.3 Flash**：最大的一个，文件 79 到 114 GB，大部分留在内存里。

## 速度有多快？

由 PolyStrata 在一台工作站上测得，用一段短对话和一个 29K token 的提示。

<table>
<tr><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **3.0BIT** | 59-66 tokens/s | 820 tokens/s |
| **UD-IQ2_XXS** | 50-52 tokens/s | 1,060-1,080 tokens/s |
| **REAP50-Q3_K_M** | 59-66 tokens/s | 1,300-1,310 tokens/s |

</td></tr>
</table>

GLM 5.3 Flash 比其他几个模型大得多，这张卡的 32 GB 显存只放得下它约五分之一的专家，所以写得慢一些。
测了什么、怎么测的：[docs/GLM.md](../GLM.md#what-it-was-measured-on)。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡。目前只在 **32 GB 显存**上跑过。放专家之前就要占掉约 12 GB，所以 12 GB 的卡启动不了。 |
| **内存** | 配 32 GB 显卡时，完整模型需要 **96 GB 或以上**，2 bit 的规格需要 82 GB，专家减半的规格需要 60 GB。显卡越小，需要的内存越多（[具体多少](#该选哪个模型)）。实际跑过的是 120 GB。 |
| **硬盘** | 模型文件 114 GB、102 GB 或 79 GB，要放在 SSD 上。 |
| **系统** | Linux，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定。它会在你的电脑上编译引擎，缺 g++ 和 CUDA 工具包时会提出帮你装。
暂时不支持：AMD 显卡、Windows 上用两张显卡、三张及以上显卡、图片。
有用户在 Windows 上用一张 24 GB 显卡加 128 GB 内存跑起来了（[报告](https://github.com/VecSzn/PolyStrata/issues/10)）。

完整列表：[docs/GLM.md](../GLM.md#what-you-need)。

## 该选哪个模型

目前有三个规格，安装程序会选内存够用的第一个。

| 规格 | 是什么 | 下载 | 配 32 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **3.0BIT** | 完整模型，约 3 bit（[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)） | 114 GB | 96 GB |
| **UD-IQ2_XXS** | 完整模型，约 2 bit（[Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)）：读提示更快，写回答更慢 | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | 去掉了一半的路由专家（patrickbdevaney 的 [REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)） | 79 GB | 60 GB |

需要多少内存取决于你的显卡，因为显卡放不下的专家都得留在内存里。

| 你的显卡 | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | 跑过的就是这台电脑，内存 120 GB |
| **24 GB** | 约 104 GB | 约 90 GB | 约 68 GB | 还没跑过 |
| **16 GB** | 约 113 GB | 约 99 GB | 约 77 GB | 还没跑过 |
| **12 GB 或以下** | | | | 启动不了 |

内存数字是安装程序根据唯一跑过的那台电脑估算的。内存不够也能启动，但回答时要从 SSD 读专家，会慢很多。
安装程序会先说明并征求你的同意。

GLM 5.3 Flash 的其他量化版本暂时不支持：[还缺什么](../GLM.md#what-is-missing)。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#glm-53-flash)。

# DeepSeek V4 Flash

返回 [README.zh-CN.md](../../README.zh-CN.md)

**DeepSeek V4 Flash**：文件 91 GB，大部分也留在内存里。它记对话占的显存很少：128K 上下文只比 32K 多占 0.7 GB。

## 速度有多快？

由 PolyStrata 在 [GLM 5.3 Flash](glm.zh-CN.md) 用的同一台工作站上测得，用一段短对话和一个 3 万 token 的提示。

<table>
<tr><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-IQ2_XXS** | 56 tokens/s | 1,040 tokens/s |

</td></tr>
</table>

同一台电脑上，llama.cpp 每秒写 25 到 28 个 token，读一个 3,384 token 的提示每秒读 96 个。
在 128K 的上下文里，一个 116,473 token 的提示读完并答完用了 174 秒；在 512K 的上下文里，一个 487,529 token 的提示用了
32 分钟。两次藏在里面的三条信息都找到了。
测了什么、怎么测的：[docs/DEEPSEEK.md](../DEEPSEEK.md#what-it-was-measured-on)。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡。目前只在 **32 GB 显存**上跑过。放专家之前就要占掉约 11.5 GB，所以 12 GB 的卡启动不了。 |
| **内存** | 配 32 GB 显卡时需要 **72 GB 或以上**，这是安装程序的估算。显卡越小，需要的内存越多（[具体多少](#该选哪个模型)）。实际跑过的是 120 GB。 |
| **硬盘** | 模型的三个文件共 91 GB，要放在 SSD 上。 |
| **系统** | Linux，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定，和 GLM 5.3 Flash 一样。还没试过：AMD 显卡、两张显卡。它只能处理文字。
有用户在 Windows 上用一张 24 GB 显卡加 128 GB 内存跑起来了（[报告](https://github.com/VecSzn/PolyStrata/issues/21)）。

完整列表：[docs/DEEPSEEK.md](../DEEPSEEK.md#what-you-need)。

## 该选哪个模型

目前只有一个规格。

| 规格 | 是什么 | 下载 | 配 32 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **UD-IQ2_XXS** | 完整模型，约 2 bit（[Unsloth](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)） | 91 GB | 72 GB |

和 GLM 5.3 Flash 一样，显卡放不下的专家都留在内存里：按安装程序的估算，配 24 GB 显卡约需 80 GB，配 16 GB 显卡约需 89 GB，
12 GB 的卡启动不了。目前只在 32 GB 的卡上试过。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#deepseek-v4-flash)。

# Flint 40B-A2.5B

返回 [README.zh-CN.md](../../README.zh-CN.md)

**Flint 40B-A2.5B**：Recursive 为编程智能体做的模型，文件 18 到 42 GB。它大部分层只往回看 513 个 token，所以长对话占的显存很少：128K 上下文只比 32K 多占 2.6 GB。

## 速度有多快？

和 [LFM2.5-24B-A2B](lfm25.zh-CN.md) 测法一样，同样两台电脑，提示约 2.9 万个 token。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、32 GB 内存</th><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、90 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 20-29 tokens/s | 1,310-1,420 tokens/s |
| **UD-Q3_K_XL** | 61-63 tokens/s | 2,000-2,010 tokens/s |
| **UD-Q6_K_XL** | 放不下 | 放不下 |
| **Q8_0** | 放不下 | 放不下 |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 251-252 tokens/s | 12,900 tokens/s |
| **UD-Q3_K_XL** | 260-264 tokens/s | 13,080-13,200 tokens/s |
| **Q8_0** | 158-160 tokens/s | 6,740-6,970 tokens/s |

</td></tr>
</table>

在笔记本上，和桌面上开着的程序一起，32 GB 内存只放得下 UD-Q3_K_XL。更大的规格要一遍遍从 SSD 重读专家，所以在那里慢，
安装程序在那里默认选 UD-Q3_K_XL。工作站的显卡能放下 UD-Q4_K_XL 和 UD-Q3_K_XL 的整个模型；UD-Q6_K_XL 没在工作站上跑。

在笔记本上，llama.cpp 用 `--fit on`（那里试过的设置里最快的）读 29,442 个 token 的提示是 390 token/s，之后写答案 35；
PolyStrata 读它是 1,950 到 2,010，写答案 49 到 52（[测量记录](../../bench/results/2026-10-24-laptop-flint-llamacpp-a37cc46/README.md)）。

在工作站上，llama.cpp 的 llama-bench 用默认的 512 批大小读 29,816 个 token 是 7,790 token/s，用 2,048 是 12,470，PolyStrata 是 12,910-12,940；
逐个 token 写答案，llama.cpp 是 222 token/s，读完那个提示之后 188，PolyStrata 是 249-256 和 212-213。在 128K 上下文里，124,122 个 token 的提示
20 秒读完并答完，藏在里面的三条信息都找到了。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡，**8 GB 或以上显存**。放专家之前要占掉约 4 GB。在笔记本的 8 GB 显卡和一张 32 GB 的卡上跑过。 |
| **内存** | 配 8 GB 显卡，UD-Q3_K_XL、UD-Q4_K_XL、UD-Q6_K_XL、Q8_0 分别约需 **22、29、40、46 GB**，这是安装程序的估算。在 32 GB 内存的笔记本上只有 UD-Q3_K_XL 留在了内存里，所以内存约 40 GB 起安装程序才默认选 UD-Q4_K_XL。32 GB 的显卡能放下 UD-Q4_K_XL 的整个模型。 |
| **硬盘** | 模型文件 18 到 42 GB。 |
| **系统** | Linux 或 Windows 10 / 11，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定，和 [Qwen3.6-35B-A3B](qwen36.zh-CN.md) 一样。还没在 AMD 显卡上试过。只能处理文字。许可证是 Apache 2.0。

完整列表：[docs/FLINT.md](../FLINT.md#what-you-need)（英文）。

## 该选哪个模型

四个规格。配 8 GB 显卡时，内存约 40 GB 起安装程序默认选 UD-Q4_K_XL，不到就选 UD-Q3_K_XL；显卡更大时 UD-Q4_K_XL 要的内存更少。
其余的要指名：`--model Q8_0`。

| 规格 | 是什么 | 下载 | 配 8 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | 完整模型，专家是 4 bit 和 5 bit（[Unsloth](https://huggingface.co/unsloth/flint-GGUF)） | 25 GB | 约 29 GB |
| **UD-Q3_K_XL** | 同上，专家是 3 bit：需要的内存最少 | 18 GB | 约 22 GB |
| **UD-Q6_K_XL** | 专家是 6 bit | 36 GB | 约 40 GB |
| **Q8_0** | 全部 8 bit，最接近训练出来的原模型 | 42 GB | 约 46 GB |

在 32 GB 内存的笔记本上，UD-Q4_K_XL 的文件没能留在内存里：短对话写答案是 20-29 token/s，UD-Q3_K_XL 是 61-63。
以 Q8_0 为参照，五个提示各写 256 个 token，其他规格选出和 Q8_0 相同的最可能 token 的步数占比：UD-Q6_K_XL 97.8%，
UD-Q4_K_XL 95.9%，UD-Q3_K_XL 94.0%。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#flint-40b-a25b)。

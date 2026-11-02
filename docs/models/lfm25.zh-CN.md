# LFM2.5-24B-A2B

返回 [README.zh-CN.md](../../README.zh-CN.md)

**LFM2.5-24B-A2B**：Liquid AI 的模型，文件 11 到 25 GB。它大部分层没有注意力，所以长对话占的显存很少：128K 上下文只比 32K 多占 2 GB。它的许可证只允许年收入低于 1,000 万美元的公司商用。

## 速度有多快？

和 [Qwen3.6-35B-A3B](qwen36.zh-CN.md) 测法一样，同样两台电脑，提示约 3 万个 token。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、32 GB 内存</th><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 70-73 tokens/s | 3,590-3,620 tokens/s |
| **UD-Q3_K_XL** | 88 tokens/s | 3,760-3,810 tokens/s |
| **QAD-Q4_0** | 77-78 tokens/s | 3,720-3,730 tokens/s |
| **UD-Q6_K_XL** | 42-43 tokens/s | 3,030-3,120 tokens/s |
| **Q8_0** | 23-34 tokens/s | 2,950-3,040 tokens/s |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 480-485 tokens/s | 23,390-23,460 tokens/s |
| **UD-Q3_K_XL** | 490-497 tokens/s | 22,920-23,400 tokens/s |
| **QAD-Q4_0** | 498 tokens/s | 23,880-23,900 tokens/s |
| **UD-Q6_K_XL** | 423-426 tokens/s | 21,250-21,270 tokens/s |
| **Q8_0** | 376 tokens/s | 23,270-23,350 tokens/s |

</td></tr>
</table>

Q8_0 在笔记本上占满了 32 GB 内存，所以在那里慢；工作站能放下每个规格的整个模型。

在笔记本上，llama.cpp b11540 用 `--n-cpu-moe 25`（那里试过的设置里最快的）读 32,000 个 token 的提示是 530 token/s，之后写答案 53；PolyStrata 读它是 3,540，写答案 71 到 73。在工作站上，llama-bench 用 llama.cpp 默认的 512 批大小读 29,816 个 token 是 13,300 token/s，用 2,048 是 23,600，和 PolyStrata 一样快；逐个 token 写答案是 455 token/s，读完那个提示之后 369，PolyStrata 是 452 和 387。在 128K 上下文里，约 124,000 个 token 的提示在工作站上 8.7 秒读完并答完，笔记本读完它要 45 秒。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡，**8 GB 或以上显存**。放专家之前要占掉约 4 GB。在笔记本的 8 GB 显卡和一张 32 GB 的卡上跑过。 |
| **内存** | 配 8 GB 显卡，UD-Q3_K_XL、QAD-Q4_0、UD-Q4_K_XL、UD-Q6_K_XL、Q8_0 分别约需 **16、18、20、27、31 GB**，这是安装程序的估算。24 GB 的显卡能放下较小三个规格的整个模型。五个规格都在一台 32 GB 内存的笔记本上跑过。 |
| **硬盘** | 模型文件 11 到 25 GB。 |
| **系统** | Linux 或 Windows 10 / 11，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定，和 [Qwen3.6-35B-A3B](qwen36.zh-CN.md) 一样。已有用户在 Linux 下的 AMD RX 6700 XT（12 GB）上跑过。只能处理文字。
Liquid AI 的许可证只允许年收入低于 1,000 万美元的公司商用（[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)）。

完整列表：[docs/LFM.md](../LFM.md#what-you-need)（英文）。

## 该选哪个模型

五个规格。安装程序默认选 UD-Q4_K_XL，内存不够时选 UD-Q3_K_XL；其余的要指名：`--model QAD-Q4_0`。

| 规格 | 是什么 | 下载 | 配 8 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | 完整模型，专家是 4 bit（[Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)） | 15 GB | 约 20 GB |
| **UD-Q3_K_XL** | 同上，专家是 3 bit：需要的内存最少 | 11 GB | 约 16 GB |
| **QAD-Q4_0** | 4 bit，[Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) 自己出的，训练时就针对这种量化，损失更小 | 14 GB | 约 18 GB |
| **UD-Q6_K_XL** | 专家是 6 bit | 22 GB | 约 27 GB |
| **Q8_0** | 全部 8 bit，最接近训练出来的原模型 | 25 GB | 约 31 GB |

以 Q8_0 为参照，五个提示各写 256 个 token，其他规格选出和 Q8_0 相同的最可能 token 的步数占比：UD-Q6_K_XL 96.8%，
UD-Q4_K_XL 95.5%，QAD-Q4_0 92.7%，UD-Q3_K_XL 91.6%。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#lfm25-24b-a2b)。

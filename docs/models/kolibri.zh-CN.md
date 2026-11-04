# Kolibri 1

返回 [README.zh-CN.md](../../README.zh-CN.md)

**Kolibri 1**：Aleph Alpha 的模型，由 19,200 个专家组成，写德语和写英语一样好，文件 29 到 48 GB，显卡小的时候大部分放在内存里。每五层里有四层只往回看 513 个 token。

## 速度有多快？

测法和 [Flint 40B-A2.5B](flint.zh-CN.md) 一样，提示长 31,676 个 token。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、32 GB 内存</th><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q4_K_M** | 没测 | 没测 |
| **Q2_K** | 没测 | 没测 |
| **Q3_K_M** | 没测 | 没测 |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q2_K** | 189-192 tokens/s | 4,980-5,070 tokens/s |

</td></tr>
</table>

工作站的显卡能放下 Q2_K 全部 19,200 个专家（25.6 GiB）。读 31,676 个 token 的提示，6.3 到 6.4 秒后开始回答，之后每秒写 129 到 131 个 token。Q4_K_M 和 Q3_K_M 没在工作站上跑过，笔记本上哪个规格都没跑过。

加入这个模型的 PR 在别的电脑上测过：RTX 3060（12 GB）、24 GB 内存、Windows，Q2_K 短聊天每秒写 21 个 token，32K 的提示每秒读 698 个；一张 Tesla V100（32 GB）上，Q4_K_M 每秒写 91 个，32K 的提示每秒读 1,017 个；两张 V100 每秒读 2,179 个。

和打了这份 GGUF 自带的 `kolibri1` 补丁的 llama.cpp 对比：照着 llama.cpp 对一段英文和一段德文提示的贪心回答走，128 步里 Q2_K 分别有 120 步和 126 步的首选 token 相同。不一样的那几步里，除了一步，llama.cpp 选的 token 都是这个引擎的第二或第三选择，几个候选几乎一样可能。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡，**8 GB 或以上显存**。放专家之前要占掉约 4 GB。在一张 12 GB 的卡、一张 32 GB 的 Tesla V100 和一张 32 GB 的 RTX 5090 上跑过。 |
| **内存** | 配 8 GB 显卡，Q4_K_M、Q2_K、Q3_K_M 分别约需 **52、33、42 GB**，这是安装程序的估算。配 32 GB 显卡，Q2_K 约需 10 GB，Q4_K_M 约需 26 GB。 |
| **硬盘** | 模型文件 29 到 48 GB。 |
| **系统** | Linux 或 Windows 10 / 11，以及 NVIDIA 的最新显卡驱动。 |

其余的安装程序都会装好，和 [Qwen3.6-35B-A3B](qwen36.zh-CN.md) 一样。还没在 AMD 显卡上试过。它只能处理文字；没有草稿模型，所以一次只写一个 token。许可证是 Apache 2.0。

## 该选哪个模型

三个规格，来自 [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF) 的转换。配 8 GB 显卡时，内存约 48 GB 起安装程序默认选 Q4_K_M，少于这个数选 Q2_K；显卡更大的话，两个规格需要的内存都更少。Q3_K_M 要指名：`--model Q3_K_M`。

| 规格 | 是什么 | 下载 | 配 8 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **Q4_K_M** | 4 bit，一半的层里 down 投影是 6 bit | 48 GB | 约 52 GB |
| **Q2_K** | 2 bit：需要的内存最少 | 29 GB | 约 33 GB |
| **Q3_K_M** | 3 bit | 38 GB | 约 42 GB |

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#kolibri-1)。

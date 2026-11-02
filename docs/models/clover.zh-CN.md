# Clover-35B-A3B-Preview

返回 [README.zh-CN.md](../../README.zh-CN.md)

**Clover-35B-A3B-Preview**：DiffuSpace 的模型，文件 13 到 21 GB。它是在 Qwen3.6-35B-A3B 上接着训练出来的，还能一次写出一整块 token。只看文字。

## 速度有多快？

由 PolyStrata 在笔记本上测得，用一段短对话和一个 3.2 万个 token 的提示。笔记本的显卡是 8 GB，显卡放不下的专家由处理器来算。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、32 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q3_K_M** | 68-73 tokens/s | 1,560-1,700 tokens/s |
| **APEX-I-NANOPLUS** | 72-76 tokens/s | 1,530-1,690 tokens/s |
| **Q4_K_M** | 38-58 tokens/s | 1,220-1,360 tokens/s |

</td></tr>
</table>

读完 3.2 万个 token 的提示之后，Q3_K_M 在笔记本上每秒写 60 到 65 个 token。Q4_K_M 会把 32 GB 内存占满，所以在那台笔记本上又慢又不稳。RTX 5090 的工作站没测。

同一台笔记本上 Qwen3.6-35B-A3B 写得更快，每秒 94 到 100 个，因为它的文件里带一个猜词块，猜下面几个词只要算一层。Clover 的文件里没有。Clover 自己也能猜一整块，但那要把整个模型再跑一遍，PolyStrata 把它关掉了：在装得下全部专家的显卡上，不猜反而更快（[测量结果](../BACKENDS.md#clover-35b-a3b-preview)，英文）。

在一张装得下 Q3_K_M 全部专家的 RTX PRO 6000 上，PolyStrata 挑的下一个 token 和 llama.cpp（它的 pull request 31306 加的 `clover_moe` 支持）一致：短提示后 128 步里有 127 步一样，5,573 个 token 的提示后 128 步里有 125 步一样。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡，**8 GB 或以上显存**。放专家之前要占掉约 3.5 GB。在笔记本的 8 GB 显卡和一张 96 GB 的显卡上跑过。还没在 AMD 显卡上试过。 |
| **内存** | 配 8 GB 显卡，APEX-I-NANOPLUS、Q3_K_M、Q4_K_M 分别约需 **16、20、25 GB**，这是安装程序的估算。24 GB 的显卡能放下较小两个规格的整个模型。三个规格都在一台 32 GB 内存的笔记本上跑过。 |
| **硬盘** | 模型文件 13、17 或 21 GB。 |
| **系统** | Linux 或 Windows 10 / 11，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定，和 [Qwen3.6-35B-A3B](qwen36.zh-CN.md) 一样。maternion 的文件里没有对话模板，安装程序会从 DiffuSpace 那里下载模型自己的模板。Clover 的许可证是 Apache 2.0。

## 该选哪个模型

三个规格，都是用 llama.cpp 的 pull request 31306 里的转换脚本做的。安装程序默认选 Q3_K_M，内存不够时选 APEX-I-NANOPLUS；Q4_K_M 要指名：`--model Q4_K_M`。

| 规格 | 是什么 | 下载 | 配 8 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **Q3_K_M** | 完整模型，专家是 3 bit（[maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)） | 17 GB | 约 20 GB |
| **APEX-I-NANOPLUS** | 专家是 2 bit 和 3 bit，按重要性矩阵量化（[IsValorum](https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF)）：需要的内存最少 | 13 GB | 约 16 GB |
| **Q4_K_M** | 专家是 4 bit（maternion） | 21 GB | 约 25 GB |

maternion 的两个规格量化时没用重要性矩阵。每个规格和原始模型差多少，还没测。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#clover-35b-a3b-preview)。

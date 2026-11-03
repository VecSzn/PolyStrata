# MiniCPM-V 4.7 35B-A3B

返回 [README.zh-CN.md](../../README.zh-CN.md)

**MiniCPM-V 4.7 35B-A3B**：OpenBMB 的模型，结构和 Qwen3.6-35B-A3B 相同，权重是自己训练的，专门做看图，文件 16 到 29 GB。

## 速度有多快？

测法和 [Qwen3.6-35B-A3B](qwen36.zh-CN.md) 相同，用的是同样的两台电脑。它的模型文件里没有预猜模块，所以一次只写一个 token。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、32 GB 内存</th><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q4_K_M** | 64 tokens/s | 1,150 tokens/s |
| **mix_q3gud_im** | 59 tokens/s | 1,130-1,150 tokens/s |
| **Q6_K** | 40 tokens/s | 960-980 tokens/s |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q4_K_M** | 308-311 tokens/s | 7,910-7,950 tokens/s |
| **mix_q3gud_im** | 291-293 tokens/s | 7,730-7,740 tokens/s |
| **Q6_K** | 277-278 tokens/s | 7,290-7,330 tokens/s |

</td></tr>
</table>

10,000 个 token 的提示，笔记本每秒读 1,260 个，工作站每秒读 8,790 个。Q6_K 在笔记本上把 32 GB 内存占满了，所以在那里慢。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡，**8 GB 或以上显存**。放专家之前要占掉约 3.5 GB。在 32 GB 的卡上和笔记本的 8 GB 显卡上跑过。 AMD 显卡在 Linux 和 Windows 上都能跑，属测试版：setup 在你电脑上编译引擎；在一张 RX 7600 上跑过。 |
| **内存** | 配 8 GB 显卡，三个规格分别约需 **19、25、31 GB**，这是安装程序的估算。32 GB 的显卡三个规格的整个模型都放得下。三个规格都在一台 32 GB 内存的笔记本上跑过。 |
| **硬盘** | 模型文件 16、21 或 29 GB，要看图片再加 1 GB。 |
| **系统** | Linux 或 Windows 10 / 11，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定，和 GLM 5.3 Flash 一样。在 Windows 上，安装程序直接从最新 release 下载编译好的引擎。
如果 CPU 不支持 AVX2，或者你这份代码的引擎在那个 release 之后又改过，
就改用 Visual Studio 的 2022 编译器和 CUDA 工具包在本机编译，缺的话会问你要不要装。暂时不支持：Windows 上用两张显卡、三张及以上显卡。在 Windows 上用 AMD 显卡时，setup 会像在 Linux 上一样编译 `strata-poly`（AMD 的 TheRock wheel 装在 `.rocm-win` 里），并警告说在那里跑还很新（[AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)）。

## 该选哪个模型

目前有三个规格，来自 OpenBMB。安装程序默认选 Q4_K_M，内存不够时选 mix_q3gud_im；Q6_K 要指名：`--model Q6_K`。

| 规格 | 是什么 | 下载 | 配 8 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **Q4_K_M** | 完整模型，专家是 4 bit（[OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)） | 21 GB | 约 25 GB |
| **mix_q3gud_im** | 同上，专家是 3 bit 和 4 bit：需要的内存最少 | 16 GB | 约 19 GB |
| **Q6_K** | 同上，专家是 6 bit | 29 GB | 约 31 GB |

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#minicpm-v-47-35b-a3b)。

# Gemma 4 26B-A4B

返回 [README.zh-CN.md](../../README.zh-CN.md)

**Gemma 4 26B-A4B**：文件 11 到 17 GB。它还能看图片。

## 速度有多快？

测法和 [Qwen3.6-35B-A3B](qwen36.zh-CN.md) 相同，用的是同样的两台电脑。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、32 GB 内存</th><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 92-93 tokens/s | 2,240 tokens/s |
| **UD-Q2_K_XL** | 101-104 tokens/s | 2,150-2,160 tokens/s |
| **UD-Q4_K_M** | 85 tokens/s | 2,150 tokens/s |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 353-356 tokens/s | 8,420-8,480 tokens/s |
| **UD-Q2_K_XL** | 332-338 tokens/s | 7,940-7,970 tokens/s |
| **UD-Q4_K_M** | 335-346 tokens/s | 8,320-8,390 tokens/s |
| **QAT-UD-Q4_K_XL** | 356-371 tokens/s | 8,460-8,680 tokens/s |

</td></tr>
</table>

10,000 个 token 的提示，笔记本每秒读 2,130 个，工作站每秒读 6,800 个。QAT-UD-Q4_K_XL 是后来用 v0.1.45-poly.4 的引擎测的；同时用这个引擎测的 UD-Q4_K_M 每秒写 333-353 个、读 8,510-8,570 个。在笔记本上，开着别的程序、内存紧张的情况下，和 UD-Q4_K_M 交替测：聊天每秒写 50-62 个，对方 47-48；长提示每秒读 1,680-1,730 个，对方 1,660-1,670；读完后每秒写 65-68 个，对方 48-49。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡，**8 GB 或以上显存**。放专家之前要占掉约 6 GB。在 32 GB 的卡上跑过，在把这张卡限制到 8 GB 的情况下跑过，也在笔记本的 8 GB 显卡上跑过。 AMD 显卡在 Linux 和 Windows 上都能跑，属测试版：setup 在你电脑上编译引擎；已有用户在 Linux 下的 RX 6700 XT（12 GB）上跑过 Gemma 4。 |
| **内存** | 配 8 GB 显卡，四个规格分别约需 **15、18、19、21 GB**，这是安装程序的估算。按同样的估算，16 GB 的显卡能放下较小两个规格和 QAT 文件的整个模型。四个规格都在一台 32 GB 内存的笔记本上跑过。 |
| **硬盘** | 模型文件和它的草稿模型共 11、13、15 或 17 GB，要看图片再加 1 GB。 |
| **系统** | Linux 或 Windows 10 / 11，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定，和 GLM 5.3 Flash 一样。在 Windows 上，安装程序直接从最新 release 下载编译好的引擎。
如果 CPU 不支持 AVX2，或者你这份代码的引擎在那个 release 之后又改过，
就改用 Visual Studio 的 2022 编译器和 CUDA 工具包在本机编译，缺的话会问你要不要装。暂时不支持：Windows 上用两张显卡、三张及以上显卡。在 Windows 上用 AMD 显卡时，setup 会像在 Linux 上一样编译 `strata-poly`（AMD 的 TheRock wheel 装在 `.rocm-win` 里），并警告说在那里跑还很新（[AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)）。

## 该选哪个模型

目前有四个规格。安装程序默认选 UD-Q3_K_M，内存不够时选 UD-Q2_K_XL；另外两个要指名：`--model UD-Q4_K_M`、`--model QAT-UD-Q4_K_XL`。

| 规格 | 是什么 | 下载 | 配 8 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | 完整模型，专家是 3 bit（[Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)），外加 Google 给它配的草稿模型 | 13 GB | 约 18 GB |
| **UD-Q2_K_XL** | 同上，专家是 2 bit：需要的内存最少 | 11 GB | 约 15 GB |
| **UD-Q4_K_M** | 同上，专家是 4 bit 和 5 bit | 17 GB | 约 21 GB |
| **QAT-UD-Q4_K_XL** | Google 量化感知训练（QAT）的权重，4 bit（[Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)），带自己的草稿模型 | 15 GB | 约 19 GB |

显卡越大，放得下的模型越多，需要的内存越少：UD-Q3_K_M 配 12 GB 显卡约 14 GB，整个模型都放进显卡之后是 10 GB。
这些是安装程序的估算，还没有在这样的电脑上验证过。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#gemma-4-26b-a4b)。

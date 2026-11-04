# Qwen3.6-35B-A3B

返回 [README.zh-CN.md](../../README.zh-CN.md)

**Qwen3.6-35B-A3B**：文件 13 到 23 GB。它还能看图片。

## 速度有多快？

由 PolyStrata 在两台电脑上测得，用一段短对话和一个 3 万到 3.2 万个 token 的提示：左边是一台 8 GB 显卡的笔记本，显卡放不下的专家由处理器来算；右边是一台工作站，显卡放得下整个模型。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、32 GB 内存</th><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 94-100 tokens/s | 1,690 tokens/s |
| **UD-Q2_K_XL** | 123-124 tokens/s | 1,590 tokens/s |
| **UD-Q4_K_M** | 82-86 tokens/s | 1,560-1,570 tokens/s |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 394-395 tokens/s | 7,800-7,810 tokens/s |
| **UD-Q2_K_XL** | 450 tokens/s | 7,720-7,740 tokens/s |
| **UD-Q4_K_M** | 395-396 tokens/s | 8,070-8,080 tokens/s |

</td></tr>
</table>

10,000 个 token 的提示，笔记本每秒读 1,660 个，工作站每秒读 6,300 个。
把工作站的显卡限制到 8 GB 的那组数字是笔记本测出来之前的替身，现在放在 [docs/BACKENDS.md](../BACKENDS.md#with-8-gb-of-vram)（英文）里。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡，**8 GB 或以上显存**。放专家之前要占掉约 5 GB。在 32 GB 的卡上跑过，在把这张卡限制到 8 GB 的情况下跑过，也在笔记本的 8 GB 显卡上跑过。 AMD 显卡在 Linux 和 Windows 上都能跑，属测试版：setup 在你电脑上编译引擎；目前只在一张 RX 7600 上用 Qwen3.6 和 Ornith 测过。 |
| **内存** | 配 8 GB 显卡，三个规格分别约需 **17、21、27 GB**，这是安装程序的估算。24 GB 的显卡能放下较小两个规格的整个模型。三个规格都在一台 32 GB 内存的笔记本上跑过。 |
| **硬盘** | 模型文件 13、17 或 23 GB，要看图片再加 1 GB。 |
| **系统** | Linux 或 Windows 10 / 11，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定，和 GLM 5.3 Flash 一样。在 Windows 上，安装程序直接从最新 release 下载编译好的引擎。
如果 CPU 不支持 AVX2，或者你这份代码的引擎在那个 release 之后又改过，
就改用 Visual Studio 的 2022 编译器和 CUDA 工具包在本机编译，缺的话会问你要不要装。暂时不支持：Windows 上用两张显卡、三张及以上显卡。在 Windows 上用 AMD 显卡时，setup 会像在 Linux 上一样编译 `strata-poly`（AMD 的 TheRock wheel 装在 `.rocm-win` 里），并警告说在那里跑还很新（[AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)）。

[Strata](https://github.com/Niko1221/Strata) 从 0.1.44 起也能用它自己的引擎跑 Qwen3.6，UD-Q4_K_XL 及以上（`--family qwen36-strata`，要 32 GB 以上内存）。
在 RTX 5090 上用 UD-Q4_K_XL 比，它读 4K 提示比 strata-poly 快 1.6 倍，读 31K 提示和写字两边差不多（[测法](../BACKENDS.md#qwen36-on-two-engines)，英文）。
更小的尺寸、8 GB 显卡还是用 strata-poly。

## 该选哪个模型

目前有三个规格。安装程序默认选 UD-Q3_K_M，内存不够时选 UD-Q2_K_XL；UD-Q4_K_M 要指名：`--model UD-Q4_K_M`。

| 规格 | 是什么 | 下载 | 配 8 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | 完整模型，专家是 3 bit（[Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)） | 17 GB | 约 21 GB |
| **UD-Q2_K_XL** | 同上，专家是 2 bit 和 3 bit：需要的内存最少 | 13 GB | 约 17 GB |
| **UD-Q4_K_M** | 同上，专家是 4 bit | 23 GB | 约 27 GB |

显卡越大，放得下的模型越多，需要的内存越少：UD-Q3_K_M 配 12 GB 显卡约 17 GB，配 16 GB 显卡约 13 GB，整个模型都放进显卡之后是 10 GB。
这些是安装程序的估算，还没有在这样的电脑上验证过。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#qwen36-35b-a3b)。

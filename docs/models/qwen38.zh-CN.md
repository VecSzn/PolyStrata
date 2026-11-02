# Qwen3.8-Flash-Next

返回 [README.zh-CN.md](../../README.zh-CN.md)

**Qwen3.8-Flash-Next**：12 GB 及以上的 NVIDIA 或 AMD 显卡，Windows 或Linux。它还能看图片。引擎是 Strata 项目的，没有改动。

## 速度有多快？

由 Strata 项目在两台普通的游戏电脑上测得，提示长 32K token。

<table>
<tr><th>NVIDIA：RTX 5070（12 GB）、Ryzen 5 7600、64 GB 内存</th><th>AMD：RX 9070 XT（16 GB）、Ryzen 9 3900X、47 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA：Q2_0 用的是引擎 0.1.36，其他各行用的是 0.1.26（4K 回答，32K 提示）。完整表格见
[DETAILS.md](../DETAILS.md#speed-measured)。显存越大的显卡越快：RTX 3090（24 GB）写回答应该能达到每秒约
100-140 个 token。长对话和其他显卡的数据：[各规格的速度](../MODELS.md#how-fast-is-each-size)、
[社区测试结果](../COMMUNITY_BENCHMARKS.md)。

## 你需要什么

| | |
| --- | --- |
| **显卡** | **NVIDIA** GeForce RTX 20、30、40 或 50 系列，或 **AMD** Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700 或 RX 6800 / 6900 系列。需要 **12 GB 或以上显存**。 |
| **内存** | 32 GB 或以上。内存大小决定能装下[哪个规格](#该选哪个模型)。64 GB 可以运行所有规格。 |
| **硬盘** | 约 80 GB 可用空间。尽量用 SSD：第一次启动会快很多。 |
| **系统** | Windows 10 / 11 或 Linux，以及 NVIDIA 或 AMD 的最新显卡驱动。 |

其他的都由安装程序搞定。两到三张显卡可以分担同一个模型（[多 GPU](../MULTI_GPU.md)）。

实验性支持，由社区成员在自己的机器上编写和测试：

- **较老的显卡**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[较老的 GPU](../OLDER_GPUS.md)。
- **Intel Arc**，在 Linux 上从源码构建：[Intel Arc](../INTEL_ARC.md)。
- **AMD Ryzen AI Max（Strix Halo）**，在 Linux 上从源码构建：[Strix Halo](../STRIX_HALO.md)。
- **不支持 AVX2 的老处理器**：能用，但很慢。[较老的 CPU](../INSTALL.md#older-cpus-experimental)。

完整列表：[docs/INSTALL.md](../INSTALL.md#what-you-need)。

## 该选哪个模型

安装程序会根据你的内存推荐一个。同一个模型有几种规格，压缩程度不同。规格越小越快，越大越聪明一些。

| 你的内存 | 选择 | 原因 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB 装得下，而且专为代码打造（如果显卡是 24 GB，Q2_0 和 IQ2_XS 也能跑） |
| **48 GB** | **IQ2_XS**（或 Q2_0，最快） | 更大的规格装不下 |
| **64 GB** | **IQ2_XS**（推荐），或 IQ3_XXS / IQ3_S | 所有规格都装得下；IQ3_S 最好，也最慢 |
| **96 GB 或以上** | **IQ3_S**，或 Unsloth 的 UD-IQ4_XS（约 4-bit） | 开着其他程序也能放下最大的规格 |

- **[Coder](../MODELS.md#coder)：** 编程版本，去掉了一半专家。它达到完整模型 SWE-bench Verified 分数的 91%（由其作者测得），
  32 GB 内存就能装下。代码以外的能力较弱，包括中文和其他中日韩文本（#438）。这类用途请选 Q2_0、IQ2_XS 或 IQ3_S，
  它们保留了所有专家。
- **[Swift 1.5](../MODELS.md#swift-15)：** 一个微调版本，回答前思考的时间短得多。你能更早拿到答案，质量基本不变。
- **[Unsloth UD-IQ4_XS](../MODELS.md#unsloth-ud-iq4_xs)：** Unsloth 的约 4-bit 版本，质量介于 IQ3_S 和 UD-Q4_K_XL 之间。
  下载 94 GB。内存少于约 80 GB 时，PolyStrata 回答时要从 SSD 读取其中一部分，所以会更慢（NVMe SSD 有帮助）。
- **[Unsloth UD-Q4_K_XL、UD-Q5_K_XL、UD-Q6_K_XL 和 Q8_0](../MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0)：** 4 到 8 位，更接近完整模型（下一个 token
  与 Q8_0 相同的比例为 93-95%）。所有专家都能放进内存时速度很快（UD-Q4_K_XL 约需 88 GB，Q8_0 约需 136 GB）；内存不够时，
  PolyStrata 回答时要从 SSD 读取一部分，会慢很多（64 GB 的电脑上 UD-Q4_K_XL 每秒 7-8.5 个 token）。
- **[OrcaRouter 的 Uncensored IQ3_XXS](../MODELS.md#orcarouter-uncensored-iq3_xxs)：** 需要手动设置，不在安装程序的菜单里。

规格、下载以及各配置能装下什么：[docs/MODELS.md](../MODELS.md)。以后想再添加模型，运行
`SETUP.bat`（Linux：`./setup.sh --setup`）。

## 它是怎么工作的？

原理图和说明：[HOW_IT_WORKS_MODELS.zh-CN.md](../HOW_IT_WORKS_MODELS.zh-CN.md#qwen38-flash-next)。

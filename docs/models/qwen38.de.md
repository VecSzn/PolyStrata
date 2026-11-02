# Qwen3.8-Flash-Next

Zurück zu [README.de.md](../../README.de.md)

**Qwen3.8-Flash-Next** auf einer NVIDIA- oder AMD-Grafikkarte ab 12 GB, unter Windows oder Linux. Es versteht auch Bilder. Seine Engine ist die von [Strata](https://github.com/Niko1221/Strata), unverändert.

## Wie schnell ist es?

Gemessen von [Strata](https://github.com/Niko1221/Strata) auf zwei ganz normalen Gaming-PCs, mit einem Prompt von 32K Tokens.

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 Tokens/s | 2,650 Tokens/s |
| **IQ2_XS** | 79 Tokens/s | 2,090 Tokens/s |
| **IQ3_XXS** | 62 Tokens/s | 1,750 Tokens/s |
| **IQ3_S** | 53 Tokens/s | 1,620 Tokens/s |
| **Coder** | 55 Tokens/s | 2,180 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 Tokens/s | 1,160 Tokens/s |
| **IQ2_XS** | 52 Tokens/s | 1,110 Tokens/s |
| **Coder** | 44 Tokens/s | 1,420 Tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 mit Engine 0.1.36, die anderen Zeilen mit 0.1.26 (4K Antworten, 32K Prompts). Die vollständigen
Tabellen stehen in [DETAILS.md](../DETAILS.md#speed-measured). Eine Karte mit mehr VRAM ist schneller: Eine
RTX 3090 (24 GB) sollte etwa 100-140 Tokens pro Sekunde schreiben. Lange Chats und andere Karten:
[Tempo jedes Modells](../MODELS.md#how-fast-is-each-size), [Ergebnisse aus der Community](../COMMUNITY_BENCHMARKS.md).

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | **NVIDIA** GeForce RTX 20, 30, 40 oder 50 Serie, oder **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 oder RX 6800 / 6900 Serie. Sie braucht **12 GB VRAM oder mehr**. |
| **RAM** | 32 GB oder mehr. Dein RAM entscheidet, [welches Modell](#welches-modell-soll-ich-nehmen) passt. Mit 64 GB laufen alle Größen. |
| **Festplatte** | Etwa 80 GB frei. Nimm wenn möglich eine SSD: Der erste Start geht dann viel schneller. |
| **System** | Windows 10 / 11 oder Linux und ein aktueller Grafiktreiber von NVIDIA oder AMD. |

Alles andere richtet der Installer ein. Zwei oder drei Karten können sich das Modell teilen ([Multi-GPU](../MULTI_GPU.md)).

Experimentell, von Community-Mitgliedern auf ihren eigenen Rechnern geschrieben und getestet:

- **Ältere Grafikkarten** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Ältere GPUs](../OLDER_GPUS.md).
- **Intel Arc**, unter Linux aus dem Quellcode gebaut: [Intel Arc](../INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, unter Linux aus dem Quellcode gebaut: [Strix Halo](../STRIX_HALO.md).
- **Ältere Prozessoren ohne AVX2**: Sie funktionieren, aber langsam. [Ältere CPUs](../INSTALL.md#older-cpus-experimental).

Die vollständige Liste: [docs/INSTALL.md](../INSTALL.md#what-you-need).

## Welches Modell soll ich nehmen?

Der Installer empfiehlt eins passend zu deinem RAM. Dasselbe Modell gibt es in mehreren Größen, mehr oder weniger
stark komprimiert. Kleinere Größen sind schneller. Größere sind etwas klüger.

| Dein RAM | Nimm | Warum |
| --- | --- | --- |
| **32 GB** | **Coder** | passt in 32 GB und ist für Code gemacht (mit einer 24-GB-Karte laufen auch Q2_0 und IQ2_XS) |
| **48 GB** | **IQ2_XS** (oder Q2_0, das schnellste) | die größeren Größen passen nicht |
| **64 GB** | **IQ2_XS** (empfohlen) oder IQ3_XXS / IQ3_S | alle Größen passen; IQ3_S ist das beste und das langsamste |
| **96 GB oder mehr** | **IQ3_S** oder Unsloths UD-IQ4_XS (~4 Bit) | Platz für die größten Größen, auch wenn alles andere offen ist |

- **[Coder](../MODELS.md#coder):** eine Version für Code, bei der die Hälfte der Experten entfernt wurde. Sie
  erreicht 91 % des SWE-bench-Verified-Werts des vollen Modells (von ihren Autoren gemessen) und passt in 32 GB RAM.
  Außerhalb von Code ist sie schwächer, auch bei chinesischem und anderem CJK-Text (#438). Nimm dafür Q2_0, IQ2_XS
  oder IQ3_S, die alle Experten behalten.
- **[Swift 1.5](../MODELS.md#swift-15):** ein Fine-Tune, das viel kürzer nachdenkt, bevor es antwortet. Du
  bekommst die Antwort früher, bei etwa gleicher Qualität.
- **[Unsloth UD-IQ4_XS](../MODELS.md#unsloth-ud-iq4_xs):** die ~4-Bit-Version von Unsloth, in der Qualität
  zwischen IQ3_S und UD-Q4_K_XL. 94 GB Download. Mit weniger als ~80 GB RAM liest PolyStrata einen Teil davon während
  der Antwort von der SSD, dort ist es also langsamer (eine NVMe-SSD hilft).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL und Q8_0](../MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):** 4 bis 8 Bit, näher am vollen Modell
  (93-95 % das gleiche nächste Token wie Q8_0). Schnell, wenn alle Experten in den RAM passen (~88 GB für UD-Q4_K_XL
  bis ~136 GB für Q8_0); mit weniger RAM liest PolyStrata einen Teil davon während der Antwort von der SSD, was viel
  langsamer ist (7-8.5 Tokens/s für UD-Q4_K_XL auf einem PC mit 64 GB).
- **[OrcaRouter's Uncensored IQ3_XXS](../MODELS.md#orcarouter-uncensored-iq3_xxs):** Das richtest du von Hand
  ein. Es steht nicht im Menü des Installers.

Größen, Downloads und was wohin passt: [docs/MODELS.md](../MODELS.md). Um später ein weiteres Modell
hinzuzufügen, starte `SETUP.bat` (Linux: `./setup.sh --setup`).

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#qwen38-flash-next).

# Ornith 1.5

Zurück zu [README.de.md](../../README.de.md)

**Ornith 1.5**, mit der Architektur von Qwen3.6-35B-A3B und anderen Gewichten: eine Datei von 14 bis 24 GB. Es versteht auch Bilder.

## Wie schnell ist es?

Gemessen wie [Qwen3.6-35B-A3B](qwen36.de.md), auf denselben zwei PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 76-82 Tokens/s | 1,620 Tokens/s |
| **APEX-I-MINI** | 80-86 Tokens/s | 1,550-1,560 Tokens/s |
| **APEX-I-QUALITY** | 59-63 Tokens/s | 1,570-1,580 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 316-319 Tokens/s | 8,040-8,110 Tokens/s |
| **APEX-I-MINI** | 289-300 Tokens/s | 7,710-7,740 Tokens/s |
| **APEX-I-QUALITY** | 297-302 Tokens/s | 7,860-7,880 Tokens/s |

</td></tr>
</table>

Ein Prompt von 10,000 Tokens wird auf dem Laptop mit 1,590 Tokens/s gelesen und auf der Workstation mit 6,400.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 4 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte, auf dieser Karte, auf 8 GB begrenzt, und auf der 8-GB-Karte eines Laptops. Auf einer AMD-Karte läuft es als Beta unter Linux und unter Windows: Setup kompiliert die Engine auf deinem PC; bisher nur mit Qwen3.6 und Ornith auf einer RX 7600 getestet. |
| **RAM** | Etwa **18, 22 oder 28 GB** neben einer 8-GB-Karte für die drei Größen, nach der Schätzung des Installers. Eine Karte mit 24 GB hält die beiden kleineren Größen ganz. Alle drei Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 14, 17 oder 24 GB für die Modelldatei, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows lädt er die Engine fertig aus dem
neuesten Release. Auf einem Prozessor ohne AVX2, oder in einer Kopie, deren Engine sich seit diesem Release geändert
hat, kompiliert er sie stattdessen mit dem 2022er Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides
zu installieren. Noch nicht: zwei Karten unter Windows, drei Karten oder mehr. Auf einer AMD-Karte unter Windows
kompiliert Setup `strata-poly` für die Karte wie unter Linux (AMDs TheRock-Wheels in `.rocm-win`) und warnt, dass
Läufe dort neu sind ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Welches Modell soll ich nehmen?

Bisher drei Größen. Der Installer nimmt APEX-I-COMPACT, auf einem PC mit weniger RAM APEX-I-MINI. APEX-I-QUALITY wählt man mit Namen: `--model APEX-I-QUALITY`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | das ganze Modell mit 3- und 4-Bit-Experten ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) von mudler) | 17 GB | etwa 22 GB |
| **APEX-I-MINI** | dasselbe mit 2- und 3-Bit-Experten: am wenigsten RAM | 14 GB | etwa 18 GB |
| **APEX-I-QUALITY** | dasselbe mit 4- bis 6-Bit-Experten | 24 GB | etwa 28 GB |

Eine größere Karte hält mehr vom Modell und braucht weniger RAM: mit APEX-I-COMPACT etwa 17 GB neben einer 12-GB-Karte, 13 GB neben einer 16-GB-Karte und 10 GB, sobald die Karte alles hält. Das sind Schätzungen des Installers. Auf einem
solchen PC wurden sie nicht geprüft.

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#ornith-15).

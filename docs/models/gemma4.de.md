# Gemma 4 26B-A4B

Zurück zu [README.de.md](../../README.de.md)

**Gemma 4 26B-A4B**, eine Datei von 11 bis 17 GB. Es versteht auch Bilder.

## Wie schnell ist es?

Gemessen wie [Qwen3.6-35B-A3B](qwen36.de.md), auf denselben zwei PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 92-93 Tokens/s | 2,240 Tokens/s |
| **UD-Q2_K_XL** | 101-104 Tokens/s | 2,150-2,160 Tokens/s |
| **UD-Q4_K_M** | 85 Tokens/s | 2,150 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 353-356 Tokens/s | 8,420-8,480 Tokens/s |
| **UD-Q2_K_XL** | 332-338 Tokens/s | 7,940-7,970 Tokens/s |
| **UD-Q4_K_M** | 335-346 Tokens/s | 8,320-8,390 Tokens/s |
| **QAT-UD-Q4_K_XL** | 356-371 Tokens/s | 8,460-8,680 Tokens/s |

</td></tr>
</table>

Ein Prompt von 10,000 Tokens wird auf dem Laptop mit 2,130 Tokens/s gelesen und auf der Workstation mit 6,800. QAT-UD-Q4_K_XL wurde später mit der Engine von v0.1.45-poly.4 gemessen; UD-Q4_K_M, daneben mit dieser Engine gemessen, schrieb 333-353 Tokens/s und las 8,510-8,570. Auf dem Laptop lief sie, mit anderen offenen Programmen und knappem RAM, abwechselnd mit UD-Q4_K_M: sie schrieb in einem Chat 50-62 Tokens/s gegen 47-48, las den langen Prompt mit 1,680-1,730 gegen 1,660-1,670 und schrieb danach 65-68 gegen 48-49.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 6 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte, auf dieser Karte, auf 8 GB begrenzt, und auf der 8-GB-Karte eines Laptops. Auf einer AMD-Karte läuft es als Beta unter Linux und unter Windows: Setup kompiliert die Engine auf deinem PC; Nutzer haben Gemma 4 unter Linux auf einer RX 6700 XT (12 GB) laufen lassen. |
| **RAM** | Etwa **15, 18, 19 oder 21 GB** neben einer 8-GB-Karte für die vier Größen, nach der Schätzung des Installers. Nach derselben Schätzung hält eine Karte mit 16 GB die beiden kleineren Größen und die QAT-Datei ganz. Alle vier Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 11, 13, 15 oder 17 GB für die Modelldatei und ihr Entwurfsmodell, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows lädt er die Engine fertig aus dem
neuesten Release. Auf einem Prozessor ohne AVX2, oder in einer Kopie, deren Engine sich seit diesem Release geändert
hat, kompiliert er sie stattdessen mit dem 2022er Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides
zu installieren. Noch nicht: zwei Karten unter Windows, drei Karten oder mehr. Auf einer AMD-Karte unter Windows
kompiliert Setup `strata-poly` für die Karte wie unter Linux (AMDs TheRock-Wheels in `.rocm-win`) und warnt, dass
Läufe dort neu sind ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Welches Modell soll ich nehmen?

Bisher vier Größen. Der Installer nimmt UD-Q3_K_M, auf einem PC mit weniger RAM UD-Q2_K_XL. Die anderen beiden wählt man mit Namen: `--model UD-Q4_K_M`, `--model QAT-UD-Q4_K_XL`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | das ganze Modell mit 3-Bit-Experten ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)) und Googles Entwurfsmodell dafür | 13 GB | etwa 18 GB |
| **UD-Q2_K_XL** | dasselbe mit 2-Bit-Experten: am wenigsten RAM | 11 GB | etwa 15 GB |
| **UD-Q4_K_M** | dasselbe mit 4- und 5-Bit-Experten | 17 GB | etwa 21 GB |
| **QAT-UD-Q4_K_XL** | Googles quantisierungsbewusst trainierte Gewichte mit 4 Bit ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)), mit eigenem Entwurfsmodell | 15 GB | etwa 19 GB |

Eine größere Karte hält mehr vom Modell und braucht weniger RAM: mit UD-Q3_K_M etwa 14 GB neben einer 12-GB-Karte und 10 GB, sobald die Karte alles hält. Das sind Schätzungen des Installers. Auf einem
solchen PC wurden sie nicht geprüft.

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#gemma-4-26b-a4b).

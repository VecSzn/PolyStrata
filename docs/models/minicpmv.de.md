# MiniCPM-V 4.7 35B-A3B

Zurück zu [README.de.md](../../README.de.md)

**MiniCPM-V 4.7 35B-A3B** von OpenBMB, mit der Architektur von Qwen3.6-35B-A3B und eigenen Gewichten, gemacht, um Bilder zu lesen: eine Datei von 16 bis 29 GB.

## Wie schnell ist es?

Gemessen wie [Qwen3.6-35B-A3B](qwen36.de.md), auf denselben zwei PCs. Seine Datei hat keinen Entwurfsblock, also schreibt es ein Token nach dem
anderen.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 64 Tokens/s | 1,150 Tokens/s |
| **mix_q3gud_im** | 59 Tokens/s | 1,130-1,150 Tokens/s |
| **Q6_K** | 40 Tokens/s | 960-980 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 308-311 Tokens/s | 7,910-7,950 Tokens/s |
| **mix_q3gud_im** | 291-293 Tokens/s | 7,730-7,740 Tokens/s |
| **Q6_K** | 277-278 Tokens/s | 7,290-7,330 Tokens/s |

</td></tr>
</table>

Ein Prompt von 10,000 Tokens wird auf dem Laptop mit 1,260 Tokens/s gelesen und auf der Workstation mit 8,790. Auf dem
Laptop füllt Q6_K die 32 GB RAM, deshalb ist es dort langsam.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 3.5 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte und auf der 8-GB-Karte eines Laptops. Auf einer AMD-Karte läuft es als Beta unter Linux und unter Windows: Setup kompiliert die Engine auf deinem PC; gelaufen ist es auf einer RX 7600. |
| **RAM** | Etwa **19, 25 oder 31 GB** neben einer 8-GB-Karte für die drei Größen, nach der Schätzung des Installers. Eine Karte mit 32 GB hält alle drei Größen ganz. Alle drei Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 16, 21 oder 29 GB für die Modelldatei, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows lädt er die Engine fertig aus dem
neuesten Release. Auf einem Prozessor ohne AVX2, oder in einer Kopie, deren Engine sich seit diesem Release geändert
hat, kompiliert er sie stattdessen mit dem 2022er Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides
zu installieren. Noch nicht: zwei Karten unter Windows, drei Karten oder mehr. Auf einer AMD-Karte unter Windows
kompiliert Setup `strata-poly` für die Karte wie unter Linux (AMDs TheRock-Wheels in `.rocm-win`) und warnt, dass
Läufe dort neu sind ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Welches Modell soll ich nehmen?

Bisher drei Größen, von OpenBMB. Der Installer nimmt Q4_K_M, auf einem PC mit weniger RAM mix_q3gud_im. Q6_K wählt man mit Namen: `--model Q6_K`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **Q4_K_M** | das ganze Modell mit 4-Bit-Experten ([OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)) | 21 GB | etwa 25 GB |
| **mix_q3gud_im** | dasselbe mit 3- und 4-Bit-Experten: am wenigsten RAM | 16 GB | etwa 19 GB |
| **Q6_K** | dasselbe mit 6-Bit-Experten | 29 GB | etwa 31 GB |

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#minicpm-v-47-35b-a3b).

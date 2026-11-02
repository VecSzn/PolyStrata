# Qwen3.6-35B-A3B

Zurück zu [README.de.md](../../README.de.md)

**Qwen3.6-35B-A3B**, eine Datei von 13 bis 23 GB. Es versteht auch Bilder.

## Wie schnell ist es?

Von PolyStrata auf zwei PCs gemessen, mit einem kurzen Chat und einem Prompt von 30,000 bis 32,000 Tokens: links ein
Laptop mit einer 8-GB-Karte, auf dem der Prozessor die Experten rechnet, für die die Karte keinen Platz hat, rechts
eine Workstation, deren Karte das ganze Modell hält.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 94-100 Tokens/s | 1,690 Tokens/s |
| **UD-Q2_K_XL** | 123-124 Tokens/s | 1,590 Tokens/s |
| **UD-Q4_K_M** | 82-86 Tokens/s | 1,560-1,570 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 394-395 Tokens/s | 7,800-7,810 Tokens/s |
| **UD-Q2_K_XL** | 450 Tokens/s | 7,720-7,740 Tokens/s |
| **UD-Q4_K_M** | 395-396 Tokens/s | 8,070-8,080 Tokens/s |

</td></tr>
</table>

Ein Prompt von 10,000 Tokens wird auf dem Laptop mit 1,660 Tokens/s gelesen und auf der Workstation mit 6,300.
Die Zahlen der auf 8 GB begrenzten
Workstation-Karte, ein Ersatz aus der Zeit vor der Messung auf dem Laptop, stehen in
[docs/BACKENDS.md](../BACKENDS.md#with-8-gb-of-vram) (englisch).

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 5 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte, auf dieser Karte, auf 8 GB begrenzt, und auf der 8-GB-Karte eines Laptops. Auf einer AMD-Karte läuft es als Beta unter Linux und unter Windows: Setup kompiliert die Engine auf deinem PC; bisher nur mit Qwen3.6 und Ornith auf einer RX 7600 getestet. |
| **RAM** | Etwa **17, 21 oder 27 GB** neben einer 8-GB-Karte für die drei Größen, nach der Schätzung des Installers. Eine Karte mit 24 GB hält die beiden kleineren Größen ganz. Alle drei Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 13, 17 oder 23 GB für die Modelldatei, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows lädt er die Engine fertig aus dem
neuesten Release. Auf einem Prozessor ohne AVX2, oder in einer Kopie, deren Engine sich seit diesem Release geändert
hat, kompiliert er sie stattdessen mit dem 2022er Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides
zu installieren. Noch nicht: zwei Karten unter Windows, drei Karten oder mehr. Auf einer AMD-Karte unter Windows
kompiliert Setup `strata-poly` für die Karte wie unter Linux (AMDs TheRock-Wheels in `.rocm-win`) und warnt, dass
Läufe dort neu sind ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

Seit 0.1.44 läuft Qwen3.6 auch auf der eigenen Engine von [Strata](https://github.com/Niko1221/Strata), ab UD-Q4_K_XL
(`--family qwen36-strata`, 32 GB RAM oder mehr). Auf einer RTX 5090 mit UD-Q4_K_XL las sie einen 4K-Prompt 1,6-mal so
schnell wie strata-poly; bei einem 31K-Prompt und beim Schreiben lagen beide etwa gleich ([gemessen](../BACKENDS.md#qwen36-on-two-engines),
auf Englisch). Die kleineren Größen für 8-GB-Karten bleiben bei strata-poly.

## Welches Modell soll ich nehmen?

Bisher drei Größen. Der Installer nimmt UD-Q3_K_M, auf einem PC mit weniger RAM UD-Q2_K_XL. UD-Q4_K_M wählt man mit Namen: `--model UD-Q4_K_M`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | das ganze Modell mit 3-Bit-Experten ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 GB | etwa 21 GB |
| **UD-Q2_K_XL** | dasselbe mit 2- und 3-Bit-Experten: am wenigsten RAM | 13 GB | etwa 17 GB |
| **UD-Q4_K_M** | dasselbe mit 4-Bit-Experten | 23 GB | etwa 27 GB |

Eine größere Karte hält mehr vom Modell und braucht weniger RAM: mit UD-Q3_K_M etwa 17 GB neben einer 12-GB-Karte, 13 GB neben einer 16-GB-Karte und 10 GB, sobald die Karte alles hält. Das sind Schätzungen des Installers. Auf einem
solchen PC wurden sie nicht geprüft.

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#qwen36-35b-a3b).

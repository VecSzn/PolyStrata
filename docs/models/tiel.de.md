# Tiel Coder 35B-A3B

Zurück zu [README.de.md](../../README.de.md)

**Tiel Coder 35B-A3B**, Ornith 1.5, von peculiar-ragdoll für Coding-Agents quantisiert: eine Datei von 14 bis 23 GB. Es versteht auch Bilder.

## Wie schnell ist es?

Gemessen wie [Qwen3.6-35B-A3B](qwen36.de.md), auf denselben zwei PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 83 Tokens/s | 1,700 Tokens/s |
| **UD-IQ3_XXS** | 96-97 Tokens/s | 1,610 Tokens/s |
| **UD-Q4_K_XL** | 74-79 Tokens/s | 1,660-1,670 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 337-345 Tokens/s | 8,080-8,110 Tokens/s |
| **UD-IQ3_XXS** | 352 Tokens/s | 7,520 Tokens/s |
| **UD-Q4_K_XL** | 316-322 Tokens/s | 7,750-7,970 Tokens/s |

</td></tr>
</table>

Ein Prompt von 10,000 Tokens wird auf dem Laptop mit 1,670 Tokens/s gelesen und auf der Workstation mit 6,300.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 5 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte, auf dieser Karte, auf 8 GB begrenzt, und auf der 8-GB-Karte eines Laptops. Auf einer AMD-Karte läuft es als Beta unter Linux und unter Windows: Setup kompiliert die Engine auf deinem PC; Nutzer haben Tiel Coder unter Linux auf einer RX 6700 XT und einer RX 7800 XT laufen lassen, unter Windows auf einer RX 9070 XT. |
| **RAM** | Etwa **18, 21 oder 27 GB** neben einer 8-GB-Karte für die drei Größen, nach der Schätzung des Installers. Eine Karte mit 24 GB hält alle drei Größen ganz. Alle drei Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 14, 17 oder 23 GB für die Modelldatei, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows lädt er die Engine fertig aus dem
neuesten Release. Auf einem Prozessor ohne AVX2, oder in einer Kopie, deren Engine sich seit diesem Release geändert
hat, kompiliert er sie stattdessen mit dem 2022er Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides
zu installieren. Noch nicht: zwei Karten unter Windows, drei Karten oder mehr. Auf einer AMD-Karte unter Windows
kompiliert Setup `strata-poly` für die Karte wie unter Linux (AMDs TheRock-Wheels in `.rocm-win`) und warnt, dass
Läufe dort neu sind ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Welches Modell soll ich nehmen?

Bisher drei Größen. Der Installer nimmt UD-Q3_K_XL, auf einem PC mit weniger RAM UD-IQ3_XXS. UD-Q4_K_XL wählt man mit Namen: `--model UD-Q4_K_XL`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-Q3_K_XL** | das ganze Modell mit 3- und 4-Bit-Experten ([peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)) | 17 GB | etwa 21 GB |
| **UD-IQ3_XXS** | dasselbe mit 2- und 3-Bit-Experten: am wenigsten RAM | 14 GB | etwa 18 GB |
| **UD-Q4_K_XL** | dasselbe mit 4- bis 6-Bit-Experten | 23 GB | etwa 27 GB |

Eine größere Karte hält mehr vom Modell und braucht weniger RAM: mit UD-Q3_K_XL etwa 17 GB neben einer 12-GB-Karte, 13 GB neben einer 16-GB-Karte und 10 GB, sobald die Karte alles hält. Das sind Schätzungen des Installers. Auf einem
solchen PC wurden sie nicht geprüft.

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#tiel-coder-35b-a3b).

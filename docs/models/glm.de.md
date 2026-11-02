# GLM 5.3 Flash

Zurück zu [README.de.md](../../README.de.md)

**GLM 5.3 Flash**, das größte: eine Datei von 79 bis 114 GB, die zum größten Teil im RAM bleibt.

## Wie schnell ist es?

Gemessen für PolyStrata auf einer Workstation, mit einem kurzen Chat und einem Prompt von 29K Tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **3.0BIT** | 59-66 Tokens/s | 820 Tokens/s |
| **UD-IQ2_XXS** | 50-52 Tokens/s | 1,060-1,080 Tokens/s |
| **REAP50-Q3_K_M** | 59-66 Tokens/s | 1,300-1,310 Tokens/s |

</td></tr>
</table>

GLM 5.3 Flash ist ein viel größeres Modell als die anderen, und nur etwa ein Fünftel seiner Experten passt in die 32 GB dieser Karte, deshalb
schreibt es langsamer. Was gemessen wurde und wie: [docs/GLM.md](../GLM.md#what-it-was-measured-on) (englisch).

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer. Bisher lief es nur mit **32 GB VRAM**. Etwa 12 GB sind belegt, bevor ein Experte geladen ist, eine 12-GB-Karte kann es also nicht starten. |
| **RAM** | Neben einer 32-GB-Karte **96 GB oder mehr** für das ganze Modell, 82 GB für seine 2-Bit-Größe, 60 GB für die Größe mit der Hälfte der Experten. Eine kleinere Karte braucht mehr RAM ([wie viel](#welches-modell-soll-ich-nehmen)). Gelaufen ist es mit 120 GB. |
| **Festplatte** | 114 GB, 102 GB oder 79 GB für die Modelldatei, auf einer SSD. |
| **System** | Linux und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein. Er kompiliert die Engine auf deinem PC und bietet an, zu installieren,
was dafür fehlt (g++ und das CUDA-Toolkit). Noch nicht: AMD-Karten, zwei Karten unter Windows, drei Karten oder mehr, Bilder.
Ein Nutzer betreibt es unter Windows mit einer 24-GB-Karte und 128 GB RAM ([Bericht](https://github.com/VecSzn/PolyStrata/issues/10)).

Die vollständige Liste: [docs/GLM.md](../GLM.md#what-you-need) (englisch).

## Welches Modell soll ich nehmen?

Bisher drei Größen. Der Installer nimmt die erste, für die dein RAM reicht.

| Größe | Was es ist | Download | RAM neben einer 32-GB-Karte |
| --- | --- | ---: | ---: |
| **3.0BIT** | das ganze Modell mit etwa 3 Bit ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 GB | 96 GB |
| **UD-IQ2_XXS** | das ganze Modell mit etwa 2 Bit ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): Es liest einen Prompt schneller und schreibt langsamer | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | die Hälfte der gerouteten Experten entfernt ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) von patrickbdevaney) | 79 GB | 60 GB |

Wie viel RAM es braucht, entscheidet deine Grafikkarte: Die Experten, die nicht auf die Karte passen, müssen im RAM bleiben.

| Deine Grafikkarte | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | der PC, auf dem es gelaufen ist (120 GB RAM) |
| **24 GB** | etwa 104 GB | etwa 90 GB | etwa 68 GB | noch nicht gelaufen |
| **16 GB** | etwa 113 GB | etwa 99 GB | etwa 77 GB | noch nicht gelaufen |
| **12 GB oder weniger** | | | | startet nicht |

Die RAM-Zahlen sind die Schätzung des Installers, von dem einen PC, auf dem es gelaufen ist. Mit weniger RAM startet es
trotzdem, liest dann aber während der Antwort Experten von der SSD und wird viel langsamer. Der Installer sagt das und
fragt vorher.

Andere Quantisierungen von GLM 5.3 Flash werden noch nicht unterstützt: [was fehlt](../GLM.md#what-is-missing) (englisch).

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#glm-53-flash).

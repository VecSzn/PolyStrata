# DeepSeek V4 Flash

Zurück zu [README.de.md](../../README.de.md)

**DeepSeek V4 Flash**, eine Datei von 91 GB, die ebenfalls zum größten Teil im RAM bleibt. Sein Gedächtnis für eine Unterhaltung ist klein: Ein Kontext von 128K braucht nur 0,7 GB mehr VRAM als 32K.

## Wie schnell ist es?

Gemessen für PolyStrata auf derselben Workstation wie [GLM 5.3 Flash](glm.de.md), mit einem kurzen Chat und einem Prompt von 30K Tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-IQ2_XXS** | 56 Tokens/s | 1,040 Tokens/s |

</td></tr>
</table>

Auf demselben PC schrieb llama.cpp 25 bis 28 Tokens pro Sekunde und las einen Prompt von 3,384 Tokens mit 96. Ein
Prompt von 116,473 Tokens in einem Kontext von 128K war nach 174 Sekunden gelesen und beantwortet, einer von 487,529
Tokens in einem Kontext von 512K nach 32 Minuten, und beide Male wurden die drei darin versteckten Fakten gefunden. Was gemessen wurde und wie: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-it-was-measured-on) (englisch).

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer. Bisher lief es nur mit **32 GB VRAM**. Etwa 11,5 GB sind belegt, bevor ein Experte geladen ist, eine 12-GB-Karte kann es also nicht starten. |
| **RAM** | Neben einer 32-GB-Karte **72 GB oder mehr**, die Schätzung des Installers. Eine kleinere Karte braucht mehr RAM ([wie viel](#welches-modell-soll-ich-nehmen)). Gelaufen ist es mit 120 GB. |
| **Festplatte** | 91 GB für die drei Dateien des Modells, auf einer SSD. |
| **System** | Linux und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Noch nicht ausprobiert: AMD-Karten, zwei
Karten. Es liest nur Text.
Ein Nutzer betreibt es unter Windows mit einer 24-GB-Karte und 128 GB RAM ([Bericht](https://github.com/VecSzn/PolyStrata/issues/21)).

Die vollständige Liste: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-you-need) (englisch).

## Welches Modell soll ich nehmen?

Bisher eine Größe.

| Größe | Was es ist | Download | RAM neben einer 32-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-IQ2_XXS** | das ganze Modell mit etwa 2 Bit ([Unsloth](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)) | 91 GB | 72 GB |

Wie bei GLM 5.3 Flash bleiben die Experten, die nicht auf die Karte passen, im RAM: nach der Schätzung des Installers
etwa 80 GB neben einer 24-GB-Karte und 89 GB neben einer 16-GB-Karte; eine 12-GB-Karte kann es nicht starten.
Ausprobiert ist nur die 32-GB-Karte.

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#deepseek-v4-flash).

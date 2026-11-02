# Flint 40B-A2.5B

Zurück zu [README.de.md](../../README.de.md)

**Flint 40B-A2.5B** von Recursive, gemacht für Coding-Agenten: eine Datei von 18 bis 42 GB. Die meisten seiner Schichten schauen nur 513 Tokens zurück, darum braucht eine lange Unterhaltung wenig VRAM: Ein Kontext von 128K braucht nur 2,6 GB mehr als 32K.

## Wie schnell ist es?

Gemessen wie [LFM2.5-24B-A2B](lfm25.de.md), auf denselben zwei PCs, mit einem Prompt von etwa 29,000 Tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 90 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 20-29 tokens/s | 1,310-1,420 tokens/s |
| **UD-Q3_K_XL** | 61-63 tokens/s | 2,000-2,010 tokens/s |
| **UD-Q6_K_XL** | Passt nicht | Passt nicht |
| **Q8_0** | Passt nicht | Passt nicht |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 251-252 tokens/s | 12,900 tokens/s |
| **UD-Q3_K_XL** | 260-264 tokens/s | 13,080-13,200 tokens/s |
| **Q8_0** | 158-160 tokens/s | 6,740-6,970 tokens/s |

</td></tr>
</table>

Auf dem Laptop bleibt nur UD-Q3_K_XL neben den Programmen des Desktops in den 32 GB RAM. Die größeren Größen lesen
ihre Experten immer wieder von der SSD, deshalb sind sie dort langsam, und der Installer nimmt dort UD-Q3_K_XL. Die
Karte der Workstation hält UD-Q4_K_XL und UD-Q3_K_XL ganz; UD-Q6_K_XL lief dort nicht.

Auf dem Laptop las llama.cpp mit `--fit on`, der schnellsten der dort ausprobierten Einstellungen, einen Prompt von
29,442 Tokens mit 390 Tokens/s und schrieb danach mit 35; PolyStrata las ihn mit 1,950 bis 2,010 und schrieb mit 49
bis 52 ([die Läufe](../../bench/results/2026-10-24-laptop-flint-llamacpp-a37cc46/README.md)).

Auf der Workstation las llama-bench von llama.cpp 29,816 Tokens mit 7,790 Tokens/s beim Standard-Batch von 512 und
mit 12,470 bei 2,048, PolyStrata mit 12,910-12,940; Token für Token schrieb llama.cpp mit 222 Tokens/s und nach diesem
Prompt mit 188, PolyStrata mit 249-256 und 212-213. Einen Prompt von 124,122 Tokens in einem Kontext von 128K las und
beantwortete die Workstation in 20 Sekunden und fand die drei darin versteckten Fakten.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 4 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf der 8-GB-Karte eines Laptops und auf einer 32-GB-Karte. |
| **RAM** | Etwa **22, 29, 40 oder 46 GB** neben einer 8-GB-Karte für UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL und Q8_0, nach der Schätzung des Installers. Auf einem Laptop mit 32 GB blieb nur UD-Q3_K_XL im RAM, darum nimmt der Installer UD-Q4_K_XL erst ab etwa 40 GB. Eine 32-GB-Karte hält UD-Q4_K_XL ganz. |
| **Festplatte** | 18 bis 42 GB für die Modelldatei. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei [Qwen3.6-35B-A3B](qwen36.de.md). Auf einer AMD-Karte wurde es noch nicht
probiert. Es liest nur Text. Seine Lizenz ist Apache 2.0.

Die ganze Liste: [docs/FLINT.md](../FLINT.md#what-you-need) (englisch).

## Welches Modell soll ich nehmen?

Vier Größen. Neben einer 8-GB-Karte nimmt der Installer UD-Q4_K_XL ab etwa 40 GB RAM und darunter UD-Q3_K_XL; neben
einer größeren Karte braucht UD-Q4_K_XL weniger. Die anderen wählt man mit Namen: `--model Q8_0`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | das ganze Modell mit 4- und 5-Bit-Experten ([Unsloth](https://huggingface.co/unsloth/flint-GGUF)) | 25 GB | etwa 29 GB |
| **UD-Q3_K_XL** | dasselbe mit 3-Bit-Experten: am wenigsten RAM | 18 GB | etwa 22 GB |
| **UD-Q6_K_XL** | 6-Bit-Experten | 36 GB | etwa 40 GB |
| **Q8_0** | durchgehend 8 Bit, am nächsten am trainierten Modell | 42 GB | etwa 46 GB |

Auf dem Laptop mit 32 GB blieb die Datei von UD-Q4_K_XL nicht im RAM: Es schrieb einen kurzen Chat mit 20-29
Tokens/s, UD-Q3_K_XL mit 61-63. Gegen Q8_0 gemessen, mit fünf Prompts zu je 256 Tokens, wählen die anderen Größen in
97,8 % der Schritte dasselbe wahrscheinlichste Token wie Q8_0 (UD-Q6_K_XL), in 95,9 % (UD-Q4_K_XL) und 94,0 %
(UD-Q3_K_XL).

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#flint-40b-a25b).

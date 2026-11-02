# LFM2.5-24B-A2B

Zurück zu [README.de.md](../../README.de.md)

**LFM2.5-24B-A2B** von Liquid AI, eine Datei von 11 bis 25 GB. Die meisten seiner Schichten haben keine Attention, darum braucht eine lange Unterhaltung wenig VRAM: Ein Kontext von 128K braucht nur 2 GB mehr als 32K. Seine Lizenz erlaubt die kommerzielle Nutzung nur Firmen mit weniger als 10 Millionen Dollar Jahresumsatz.

## Wie schnell ist es?

Gemessen wie [Qwen3.6-35B-A3B](qwen36.de.md), auf denselben zwei PCs, mit einem Prompt von etwa 30,000 Tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 70-73 Tokens/s | 3,590-3,620 Tokens/s |
| **UD-Q3_K_XL** | 88 Tokens/s | 3,760-3,810 Tokens/s |
| **QAD-Q4_0** | 77-78 Tokens/s | 3,720-3,730 Tokens/s |
| **UD-Q6_K_XL** | 42-43 Tokens/s | 3,030-3,120 Tokens/s |
| **Q8_0** | 23-34 Tokens/s | 2,950-3,040 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 480-485 Tokens/s | 23,390-23,460 Tokens/s |
| **UD-Q3_K_XL** | 490-497 Tokens/s | 22,920-23,400 Tokens/s |
| **QAD-Q4_0** | 498 Tokens/s | 23,880-23,900 Tokens/s |
| **UD-Q6_K_XL** | 423-426 Tokens/s | 21,250-21,270 Tokens/s |
| **Q8_0** | 376 Tokens/s | 23,270-23,350 Tokens/s |

</td></tr>
</table>

Auf dem Laptop füllt Q8_0 die 32 GB RAM, deshalb ist es dort langsam; die Workstation hält jede Größe ganz.

Auf dem Laptop las llama.cpp b11540 mit `--n-cpu-moe 25`, der besten der dort ausprobierten Einstellungen, einen
Prompt von 32,000 Tokens mit 530 Tokens/s und schrieb danach mit 53; PolyStrata las ihn mit 3,540 und schrieb mit 71
bis 73. Auf der Workstation las llama-bench 29,816 Tokens mit 13,300 Tokens/s bei llama.cpps Standard-Batch von 512
und mit 23,600 bei 2,048, so schnell wie PolyStrata, und schrieb Token für Token mit 455 Tokens/s, nach diesem Prompt
mit 369; PolyStrata schreibt mit 452 und 387. Einen Prompt von etwa 124,000 Tokens in einem Kontext von 128K las und
beantwortete die Workstation in 8,7 Sekunden; der Laptop las ihn in 45 Sekunden.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 4 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf der 8-GB-Karte eines Laptops und auf einer 32-GB-Karte. |
| **RAM** | Etwa **16, 18, 20, 27 oder 31 GB** neben einer 8-GB-Karte für UD-Q3_K_XL, QAD-Q4_0, UD-Q4_K_XL, UD-Q6_K_XL und Q8_0, nach der Schätzung des Installers. Eine Karte mit 24 GB hält die drei kleineren Größen ganz. Alle fünf liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 11 bis 25 GB für die Modelldatei. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei [Qwen3.6-35B-A3B](qwen36.de.md). Ein Nutzer hat es unter Linux auf einer
AMD RX 6700 XT (12 GB) laufen lassen. Es liest nur Text. Die Lizenz von Liquid AI erlaubt die kommerzielle Nutzung nur Firmen mit weniger als
10 Millionen Dollar Jahresumsatz ([LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)).

Die ganze Liste: [docs/LFM.md](../LFM.md#what-you-need) (englisch).

## Welches Modell soll ich nehmen?

Fünf Größen. Der Installer nimmt UD-Q4_K_XL, auf einem PC mit weniger RAM UD-Q3_K_XL. Die anderen wählt man mit Namen: `--model QAD-Q4_0`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | das ganze Modell mit 4-Bit-Experten ([Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)) | 15 GB | etwa 20 GB |
| **UD-Q3_K_XL** | dasselbe mit 3-Bit-Experten: am wenigsten RAM | 11 GB | etwa 16 GB |
| **QAD-Q4_0** | 4 Bit, von [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) selbst, so trainiert, dass es durch diese Quantisierung weniger verliert | 14 GB | etwa 18 GB |
| **UD-Q6_K_XL** | 6-Bit-Experten | 22 GB | etwa 27 GB |
| **Q8_0** | durchgehend 8 Bit, am nächsten am trainierten Modell | 25 GB | etwa 31 GB |

Gegen Q8_0 gemessen, mit fünf Prompts zu je 256 Tokens, wählen die anderen Größen in 96,8 % der Schritte dasselbe
wahrscheinlichste Token wie Q8_0 (UD-Q6_K_XL), in 95,5 % (UD-Q4_K_XL), 92,7 % (QAD-Q4_0) und 91,6 % (UD-Q3_K_XL).

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#lfm25-24b-a2b).

# Kolibri 1

Zurück zu [README.de.md](../../README.de.md)

**Kolibri 1** von Aleph Alpha, ein Modell aus 19,200 Experten, das Deutsch so gut schreibt wie Englisch: eine Datei von
29 bis 48 GB, die neben einer kleinen Karte größtenteils im RAM bleibt. Vier von fünf Schichten schauen nur 513 Tokens
zurück.

## Wie schnell ist es?

Gemessen wie [Flint 40B-A2.5B](flint.de.md), mit einem Prompt von 31,676 Tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q4_K_M** | Nicht gemessen | Nicht gemessen |
| **Q2_K** | Nicht gemessen | Nicht gemessen |
| **Q3_K_M** | Nicht gemessen | Nicht gemessen |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q2_K** | 189-192 tokens/s | 4,980-5,070 tokens/s |

</td></tr>
</table>

Die Karte der Workstation hält alle 19,200 Experten von Q2_K (25,6 GiB). Den Prompt von 31,676 Tokens begann es nach
6,3 bis 6,4 Sekunden zu beantworten und schrieb danach mit 129-131 Tokens/s. Q4_K_M und Q3_K_M liefen dort nicht, und
auf dem Laptop lief keine Größe.

Der Pull Request, der das Modell hinzufügte, hat es auf anderen PCs gemessen. Auf einer RTX 3060 (12 GB) mit 24 GB RAM
unter Windows schrieb Q2_K einen kurzen Chat mit 21 Tokens/s und las einen 32K-Prompt mit 698. Auf einer Tesla V100
(32 GB) schrieb Q4_K_M mit 91 Tokens/s und las einen 32K-Prompt mit 1,017; auf zweien las es mit 2,179.

Gegen llama.cpp mit dem `kolibri1`-Patch der Umwandlung, entlang seiner Greedy-Antwort auf einen englischen und einen
deutschen Prompt, wählt Q2_K bei 120 und 126 von 128 Schritten dasselbe wahrscheinlichste Token. Wo sie sich
unterscheiden, ist das Token von llama.cpp bis auf einen Schritt die zweite oder dritte Wahl dieser Engine, fast
gleich wahrscheinlich.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 4 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 12-GB-Karte, einer Tesla V100 mit 32 GB und einer RTX 5090 mit 32 GB. |
| **RAM** | Etwa **52, 33 oder 42 GB** neben einer 8-GB-Karte für Q4_K_M, Q2_K und Q3_K_M, nach der Schätzung des Installers. Neben einer 32-GB-Karte braucht Q2_K etwa 10 GB und Q4_K_M etwa 26 GB. |
| **Festplatte** | 29 bis 48 GB für die Modelldatei. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei [Qwen3.6-35B-A3B](qwen36.de.md). Auf einer AMD-Karte wurde es noch nicht
ausprobiert. Es liest nur Text und hat kein Entwurfsmodell, schreibt also ein Token nach dem anderen. Seine Lizenz ist
Apache 2.0.

## Welches Modell soll ich nehmen?

Drei Größen aus der Umwandlung von [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF). Neben einer 8-GB-Karte
nimmt der Installer Q4_K_M ab etwa 48 GB RAM und darunter Q2_K; neben einer größeren Karte brauchen beide weniger.
Q3_K_M wählt man mit Namen: `--model Q3_K_M`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **Q4_K_M** | 4 Bit, mit 6-Bit-Down-Projektionen in der Hälfte der Schichten | 48 GB | etwa 52 GB |
| **Q2_K** | 2 Bit: am wenigsten RAM | 29 GB | etwa 33 GB |
| **Q3_K_M** | 3 Bit | 38 GB | etwa 42 GB |

## Wie funktioniert es?

Bild und Erklärung: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#kolibri-1).

# Clover-35B-A3B-Preview

Zurück zu [README.de.md](../../README.de.md)

**Clover-35B-A3B-Preview** von DiffuSpace, eine Datei von 13 bis 21 GB. Es ist Qwen3.6-35B-A3B, weiter trainiert, damit
es auch einen Block von Tokens auf einmal schreibt. Es liest nur Text.

## Wie schnell ist es?

Von PolyStrata auf dem Laptop gemessen, mit einem kurzen Chat und einem Prompt von 32,000 Tokens. Seine Karte hat
8 GB, und der Prozessor rechnet die Experten, für die die Karte keinen Platz hat.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q3_K_M** | 68-73 Tokens/s | 1,560-1,700 Tokens/s |
| **APEX-I-NANOPLUS** | 72-76 Tokens/s | 1,530-1,690 Tokens/s |
| **Q4_K_M** | 38-58 Tokens/s | 1,220-1,360 Tokens/s |

</td></tr>
</table>

Nach dem Prompt von 32,000 Tokens schreibt der Laptop mit Q3_K_M 60 bis 65 Tokens/s. Q4_K_M füllt dort die 32 GB RAM,
und das macht es langsam und ungleichmäßig. Die Workstation mit der RTX 5090 wurde nicht gemessen.

Qwen3.6-35B-A3B schreibt auf demselben Laptop schneller, 94 bis 100 Tokens/s, weil seine Datei einen Draft-Block hat,
der die nächsten Wörter mit der Arbeit einer Schicht rät. Clovers Dateien haben keinen. Clover kann selbst einen Block
von Wörtern raten, aber dafür läuft das ganze Modell noch einmal, und PolyStrata lässt das aus: Auf einer Karte, die
jeden Experten hielt, kamen die Antworten ohne es früher ([gemessen](../BACKENDS.md#clover-35b-a3b-preview), auf
Englisch).

Auf einer RTX PRO 6000, die jeden Experten von Q3_K_M hielt, wählte PolyStrata nach einem kurzen Prompt bei 127 von
128 Schritten dasselbe nächste Token wie llama.cpp (die `clover_moe`-Unterstützung seines Pull Requests 31306), nach
5,573 Tokens bei 125 von 128.

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 3,5 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf der 8-GB-Karte eines Laptops und auf einer 96-GB-Karte. Auf einer AMD-Karte wurde es noch nicht probiert. |
| **RAM** | Etwa **16, 20 oder 25 GB** neben einer 8-GB-Karte für APEX-I-NANOPLUS, Q3_K_M und Q4_K_M, nach der Schätzung des Installers. Eine Karte mit 24 GB hält die beiden kleineren Größen ganz. Alle drei liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 13, 17 oder 21 GB für die Modelldatei. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Den Rest richtet der Installer ein, wie für [Qwen3.6-35B-A3B](qwen36.de.md). maternions Dateien haben keine
Chat-Vorlage, deshalb lädt der Installer die eigene Vorlage des Modells von DiffuSpace herunter. Clovers Lizenz ist
Apache 2.0.

## Welches Modell soll ich nehmen?

Drei Größen, alle mit dem Konverter aus dem Pull Request 31306 von llama.cpp gemacht. Der Installer nimmt Q3_K_M, auf
einem PC mit weniger RAM APEX-I-NANOPLUS. Q4_K_M wählt man mit Namen: `--model Q4_K_M`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **Q3_K_M** | das ganze Modell mit 3-Bit-Experten ([maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)) | 17 GB | etwa 20 GB |
| **APEX-I-NANOPLUS** | 2- und 3-Bit-Experten aus einer Importance-Matrix ([IsValorum](https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF)): am wenigsten RAM | 13 GB | etwa 16 GB |
| **Q4_K_M** | 4-Bit-Experten (maternion) | 21 GB | etwa 25 GB |

maternions zwei Größen wurden ohne Importance-Matrix quantisiert. Wie nah jede Größe dem trainierten Modell kommt,
wurde nicht gemessen.

## Wie funktioniert es?

Das Diagramm und wie es funktioniert: [HOW_IT_WORKS_MODELS.de.md](../HOW_IT_WORKS_MODELS.de.md#clover-35b-a3b-preview).

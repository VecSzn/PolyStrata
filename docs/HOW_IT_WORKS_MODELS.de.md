# Wie funktioniert es?

Zurück zu [README.de.md](../README.de.md)

Modelle wie dieses laufen normalerweise auf Servern mit Hunderten Gigabyte Grafikspeicher. Deine Grafikkarte hat
8-32 GB. PolyStrata bringt das Modell trotzdem unter, indem es **die Arbeit auf deinen ganzen PC verteilt**. Stell dir
eine Küche vor: Was du ständig brauchst, bleibt auf der Arbeitsfläche, der Rest wartet in der Vorratskammer.

## Qwen3.8-Flash-Next

<p align="center"><img src="media/how-it-works.svg" width="860" alt="Die 24,576 Experten des Modells: die meistgenutzten auf der Grafikkarte, alle im RAM, eine Nachschlagetabelle auf der SSD"></p>

- **Das Modell ist ein Team aus 24,576 kleinen Spezialisten („Experten“).** Jedes Wort braucht nur 10 davon.
- **Deine Grafikkarte** hält die paar tausend Experten, die am häufigsten gebraucht werden. **Dein RAM** hält alle,
  und **dein Prozessor** rechnet gleichzeitig am Rest. **Deine SSD** hält eine große Nachschlagetabelle.

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="Ein kleiner Helfer rät die nächsten Wörter; das große Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Ein kleiner Helfer rät die nächsten paar Wörter. Das große Modell prüft sie alle auf
  einmal. Du bekommst dieselbe Antwort, nur 1.6-1.8x schneller.
- **Lange Texte werden in großen Stücken gelesen** (bis zu 8,192 Tokens auf einmal), mit über 1,000 Tokens pro
  Sekunde.

Die längere Erklärung: [docs/HOW_IT_WORKS.md](HOW_IT_WORKS.md). Jeder Teil und seine Zahlen:
[die Details](DETAILS.md#how-it-works) und das [Paper](paper/Strata-Paper.pdf).

## GLM 5.3 Flash

<p align="center"><img src="media/how-it-works-glm.svg" width="860" alt="Die 12,096 Experten des Modells: die meistgenutzten auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 12,096 kleinen Experten**, 288 in jeder von 42 Schichten. Jedes Wort braucht nur 8 aus
  jeder Schicht.
- **Deine Grafikkarte** hält die am häufigsten gebrauchten, etwa 2,470 auf einer 32-GB-Karte: Die Engine zählt, welche
  deine Arbeit braucht, und wählt beim nächsten Start neu. **Dein RAM** hält die übrigen, und **dein Prozessor**
  rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne zweite
  Kopie; ein Start dauert etwa 6 Sekunden, sobald die Datei in deinem RAM ist.

<p align="center"><img src="media/guess-and-check-glm.svg" width="860" alt="Ein Block des Modells rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Ein Block des Modells, der zum Raten gemacht ist, schlägt die nächsten paar Wörter vor. Das
  Modell prüft sie alle auf einmal. Du bekommst dieselbe Antwort, nur 1.1-1.2x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal; die Experten, die nicht auf der Karte
  liegen, laufen durch sie), mit 780 bis 1,250 Tokens pro Sekunde.

Jeder Teil und seine Zahlen: [docs/GLM.md](GLM.md#how-it-uses-the-card) (englisch).

## DeepSeek V4 Flash

<p align="center"><img src="media/how-it-works-dsv4.svg" width="860" alt="Die 11,008 Experten des Modells: die meistbenutzten auf der Grafikkarte, der Rest im RAM, die Dateien des Modells auf der SSD"></p>

- **Das Modell ist ein Team aus 11,008 kleinen Experten**, 256 in jeder von 43 Schichten. Jedes Wort braucht nur 6
  aus jeder Schicht, dazu einen, den jedes Wort benutzt.
- **Deine Grafikkarte** behält die meistbenutzten, etwa 2,840 auf einer 32-GB-Karte: Die Engine zählt, welche deine
  Arbeit braucht. **Dein RAM** hält den Rest, und **dein Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die
  Dateien des Modells, die dort benutzt werden, wo sie liegen, ohne zweite Kopie.
- **Ein kleines Gedächtnis für die Unterhaltung:** Ein Wort liest die letzten 128 Wörter ganz und den Rest als
  komprimierte Zeilen, eine für je 4 oder je 128 Wörter. Ein Token Kontext braucht 6,880 Bytes, also lässt auch ein
  langer Kontext der Karte fast den ganzen Platz für Experten.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal), 30,000 Tokens mit etwa 1,000 Tokens pro
  Sekunde. Raten und Prüfen gibt es noch nicht: Das Entwurfsmodell, das es dafür gibt, läuft noch nicht.

Jeder Teil und seine Zahlen: [docs/DEEPSEEK.md](DEEPSEEK.md#how-it-differs-from-the-other-models) (englisch).

## LFM2.5-24B-A2B

<p align="center"><img src="media/how-it-works-lfm25.svg" width="860" alt="Die 2,432 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 2,432 kleinen Experten**, 64 in jeder von 38 Schichten. Jedes Wort braucht nur 4 aus
  jeder Schicht.
- **Die meisten seiner 40 Schichten haben keine Attention.** 30 davon mischen die Wörter mit einer kurzen Faltung über
  die letzten drei und merken sich zwei Vektoren je Schicht. Nur die übrigen 10 behalten die ganze Unterhaltung, darum
  ist sein Gedächtnis für eine Unterhaltung klein: Ein Kontext von 128K braucht nur 2 GB mehr VRAM als 32K.
- **Deine Grafikkarte** hält so viele Experten, wie passen: etwa 900 bis 1,300 auf einer 8-GB-Karte, ab 24 GB alle bei
  den 4-Bit-Größen. Auf der 8-GB-Karte liefern die Experten der Karte zwei Drittel bis vier Fünftel dessen, was eine
  Antwort braucht. **Dein RAM** hält die übrigen, und **dein Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält
  die Modelldatei, die dort benutzt wird, wo sie liegt, ohne zweite Kopie.
- **Kein Entwurfsblock.** Das Modell hat nichts, was vorausrät, darum rät die Engine nur aus Text, den die
  Unterhaltung wiederholt.
- **Wenn die Karte alle Experten hält**, bleibt ein ganzer Schritt auf der Karte: Sie wählt die Experten der Wörter
  selbst und lässt die 40 Schichten eines Tokens als einen aufgezeichneten Start laufen.

Jeder Teil und seine Zahlen: [docs/LFM.md](LFM.md) (englisch).

## Flint 40B-A2.5B

<p align="center"><img src="media/how-it-works-flint.svg" width="860" alt="Die 12,032 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 12,032 kleinen Experten**, 256 in jeder von 47 Schichten. Jedes Wort braucht nur 6 aus
  jeder Schicht und einen gemeinsamen Experten, den jedes Wort benutzt.
- **Die meisten seiner 49 Schichten schauen 513 Tokens zurück.** 35 davon achten nur auf die letzten 513 Tokens und
  halten einen Ring von Zeilen, der nicht wächst. Nur die übrigen 14 behalten die ganze Unterhaltung, darum ist sein
  Gedächtnis für eine Unterhaltung klein: Ein Kontext von 128K braucht nur 2,6 GB mehr VRAM als 32K.
- **Deine Grafikkarte** hält so viele Experten, wie passen: etwa 1,600 bis 2,500 auf einer 8-GB-Karte, alle von
  UD-Q4_K_XL auf einer 32-GB-Karte. **Dein RAM** hält die übrigen, und **dein Prozessor** rechnet sie gleichzeitig.
  **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne zweite Kopie.
- **Kein Entwurfsblock.** Das Modell hat nichts, was vorausrät, darum rät die Engine nur aus Text, den die
  Unterhaltung wiederholt.

Jeder Teil und seine Zahlen: [docs/FLINT.md](FLINT.md) (englisch).

## Qwen3.6-35B-A3B

<p align="center"><img src="media/how-it-works-qwen36.svg" width="860" alt="Die 10,240 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 10,240 kleinen Experten**, 256 in jeder von 40 Schichten. Jedes Wort braucht nur 8 aus
  jeder Schicht.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 2,600 auf einer 8-GB-Karte, ab 24 GB alle. Während es
  schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und auf der 8-GB-Karte
  liefern die Experten der Karte etwa 60 % dessen, was eine Antwort braucht. **Dein RAM** hält die übrigen, und **dein
  Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne
  zweite Kopie.

<p align="center"><img src="media/guess-and-check-qwen36.svg" width="860" alt="Der Entwurfsblock des Modells rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Der eigene Entwurfsblock des Modells, der in seiner Datei liegt, schlägt die nächsten paar
  Wörter vor. Das Modell prüft sie alle auf einmal. Du bekommst dieselbe Antwort, nur 1.2-1.5x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (englisch).

## Clover-35B-A3B-Preview

<p align="center"><img src="media/how-it-works-clover.svg" width="860" alt="Die 10,240 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Qwen3.6-35B-A3B, weiter trainiert:** ein Team aus 10,240 kleinen Experten, 256 in jeder von 40 Schichten, und nur
  8 aus jeder Schicht für jedes Wort.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 2,200 bis 4,100 auf einer 8-GB-Karte, umso mehr, je kleiner die
  Größe, und ab 24 GB alle von Q3_K_M. Auf der 8-GB-Karte liefern die Experten der Karte 60 bis 80 % dessen, was eine
  Antwort braucht. **Dein RAM** hält die übrigen, und **dein Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält
  die Modelldatei, die dort benutzt wird, wo sie liegt, ohne zweite Kopie.
- **Kein Entwurfsblock.** Clover wurde darauf trainiert, einen Block von Wörtern auf einmal zu schreiben, indem es
  Lücken gemeinsam füllt, und kann so auch die nächsten Wörter raten. Dafür läuft für jeden geratenen Block das ganze
  Modell noch einmal, und das kostete mehr, als die richtigen Treffer sparten, deshalb ist es aus
  ([gemessen](BACKENDS.md#clover-35b-a3b-preview), englisch). Die Engine rät nur aus Text, den das Gespräch wiederholt.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal), wie bei Qwen3.6-35B-A3B.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](BACKENDS.md#clover-35b-a3b-preview) (englisch).

## Ornith 1.5

<p align="center"><img src="media/how-it-works-ornith.svg" width="860" alt="Die 10,240 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Die Architektur von Qwen3.6-35B-A3B mit anderen Gewichten:** ein Team aus 10,240 kleinen Experten, 256 in jeder
  von 40 Schichten, und nur 8 aus jeder Schicht für jedes Wort.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 3,100 auf einer 8-GB-Karte, ab 24 GB alle. Während es
  schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und auf der 8-GB-Karte
  liefern die Experten der Karte etwa 60 % dessen, was eine Antwort braucht. **Dein RAM** hält die übrigen, und **dein
  Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne
  zweite Kopie.

<p align="center"><img src="media/guess-and-check-ornith.svg" width="860" alt="Der Entwurfsblock des Modells rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Auch sein Entwurfsblock liegt in seiner Datei und schlägt die nächsten paar Wörter vor. Er
  rät seltener richtig als der von Qwen3.6-35B-A3B, also bekommst du dieselbe Antwort 1.1-1.2x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (englisch).

## Gemma 4 26B-A4B

<p align="center"><img src="media/how-it-works-gemma4.svg" width="860" alt="Die 3,840 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 3,840 kleinen Experten**, 128 in jeder von 30 Schichten. Jedes Wort braucht nur 8 aus
  jeder Schicht, neben einem Teil der Schicht, den jedes Wort braucht.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 1,000 auf einer 8-GB-Karte, nach der Schätzung des Installers ab
  16 GB alle. Während es schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und
  auf der 8-GB-Karte liefern die Experten der Karte etwa 55 % dessen, was eine Antwort braucht. **Dein RAM** hält die
  übrigen, und **dein Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei und ihr Entwurfsmodell,
  die dort benutzt werden, wo sie liegen, ohne zweite Kopie.

<p align="center"><img src="media/guess-and-check-gemma4.svg" width="860" alt="Googles kleines Entwurfsmodell rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Googles kleines Entwurfsmodell dafür, eine Datei von 0.5 GB, die der Installer mit dem
  Modell herunterlädt, schlägt die nächsten paar Wörter vor. Das Modell prüft sie alle auf einmal. Du bekommst
  dieselbe Antwort, nur 1.3-1.6x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (englisch).

## Tiel Coder 35B-A3B

<p align="center"><img src="media/how-it-works-tiel.svg" width="860" alt="Die 10,240 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Ornith 1.5 in einer anderen Quantisierung:** ein Team aus 10,240 kleinen Experten, 256 in jeder
  von 40 Schichten, und nur 8 aus jeder Schicht für jedes Wort.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 2,700 auf einer 8-GB-Karte, ab 24 GB alle. Während es
  schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und auf der 8-GB-Karte
  liefern die Experten der Karte etwa 50 % dessen, was eine Antwort braucht. **Dein RAM** hält die übrigen, und **dein
  Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne
  zweite Kopie.

<p align="center"><img src="media/guess-and-check-tiel.svg" width="860" alt="Der Entwurfsblock des Modells rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Auch sein Entwurfsblock liegt in seiner Datei und schlägt die nächsten paar Wörter vor. Er
  rät seltener richtig als der von Qwen3.6-35B-A3B, also bekommst du dieselbe Antwort 1.2-1.3x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (englisch).

## MiniCPM-V 4.7 35B-A3B

<p align="center"><img src="media/how-it-works-minicpmv.svg" width="860" alt="Die 10,240 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Die Architektur von Qwen3.6-35B-A3B mit den Gewichten von OpenBMB:** ein Team aus 10,240 kleinen Experten, 256 in
  jeder von 40 Schichten, und nur 8 aus jeder Schicht für jedes Wort.
- **Deine Grafikkarte** hält so viele, wie passen: mit Q4_K_M etwa 2,000 auf einer 8-GB-Karte, auf einer 32-GB-Karte
  alle. Während es schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und auf der
  8-GB-Karte liefern die Experten der Karte etwa 60 bis 70 % dessen, was eine Antwort braucht. **Dein RAM** hält die
  übrigen, und **dein Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird,
  wo sie liegt, ohne zweite Kopie.
- **Ein Wort nach dem anderen:** Seine Datei hat keinen Entwurfsblock, also rät nichts voraus.
- **Bilder:** Sein eigener Bild-Encoder liest ein großes Bild als Übersicht und einige Kacheln. Ein Screenshot von 1,200
  mal 398 Pixeln wird zu etwa 300 Tokens, ein Diagramm von 2,520 mal 1,620 zu etwa 600.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (englisch).

## Kolibri 1

<p align="center"><img src="media/how-it-works-kolibri.svg" width="860" alt="Die 19,200 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 19,200 kleinen Experten**, 384 in jeder von 50 Schichten. Jedes Wort braucht nur 6 aus
  jeder Schicht und einen gemeinsamen Experten, den jedes Wort benutzt.
- **Vier von fünf Schichten schauen 513 Tokens zurück.** Sie halten einen Ring von Zeilen, der nicht wächst. Die
  fünfte behält die ganze Unterhaltung und kennt überhaupt keine Wortpositionen.
- **Deine Grafikkarte** hält so viele Experten, wie passen: etwa ein Zehntel davon auf einer 8-GB-Karte, alle von Q2_K
  auf einer 32-GB-Karte. **Dein RAM** hält die übrigen, und **dein Prozessor** rechnet sie gleichzeitig. **Deine SSD**
  hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne zweite Kopie.
- **Kein Entwurfsblock.** Das Modell hat nichts, was vorausrät, darum rät die Engine nur aus Text, den die
  Unterhaltung wiederholt.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (englisch).

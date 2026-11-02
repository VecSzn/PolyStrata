# Comment ça marche ?

Retour à [README.fr.md](../README.fr.md)

Les modèles comme celui-ci tournent d'habitude sur des serveurs avec des centaines de gigaoctets de mémoire
graphique. Votre carte graphique a 8-32 Go. PolyStrata fait tenir le modèle en **répartissant le travail sur tout votre
PC**. Pensez à une cuisine : ce que vous utilisez tout le temps reste sur le plan de travail, et le reste attend dans
le placard.

## Qwen3.8-Flash-Next

<p align="center"><img src="media/how-it-works.svg" width="860" alt="Les 24,576 experts du modèle : les plus sollicités sur la carte graphique, tous en RAM, une table de correspondance sur le SSD"></p>

- **Le modèle est une équipe de 24,576 petits spécialistes (les « experts »).** Chaque mot n'en demande que 10.
- **Votre carte graphique** garde les quelques milliers d'experts les plus utilisés. **Votre RAM** les contient tous,
  et **votre processeur** travaille sur les autres en même temps. **Votre SSD** contient une grande table de
  correspondance.

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="Un petit assistant devine les mots suivants ; le grand modèle les vérifie tous d'un coup et garde les bons"></p>

- **Deviner, puis vérifier :** un petit assistant devine les quelques mots suivants. Le grand modèle les vérifie
  tous d'un coup. Vous obtenez la même réponse, 1.6-1.8x plus tôt.
- **Les longs textes sont lus par gros morceaux** (jusqu'à 8,192 tokens à la fois), à plus de 1,000 tokens par seconde.

L'explication plus longue : [docs/HOW_IT_WORKS.md](HOW_IT_WORKS.md). Chaque partie et ses chiffres :
[les détails](DETAILS.md#how-it-works) et l'[article](paper/Strata-Paper.pdf).

## GLM 5.3 Flash

<p align="center"><img src="media/how-it-works-glm.svg" width="860" alt="Les 12,096 experts du modèle : les plus utilisés sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Le modèle est une équipe de 12,096 petits experts**, 288 dans chacune de 42 couches. Chaque mot n'en demande que 8
  par couche.
- **Votre carte graphique** garde les plus utilisés, environ 2,470 sur une carte de 32 Go : le moteur compte ceux que
  votre travail utilise et refait son choix au démarrage suivant. **Votre RAM** contient les autres, et **votre
  processeur** les calcule en même temps. **Votre SSD** contient le fichier du modèle, utilisé là où il se trouve,
  sans seconde copie ; un démarrage prend environ 6 secondes une fois le fichier dans votre RAM.

<p align="center"><img src="media/guess-and-check-glm.svg" width="860" alt="Un bloc du modèle devine les mots suivants ; le modèle les vérifie tous d'un coup et garde les bons"></p>

- **Deviner, puis vérifier :** un bloc du modèle fait pour deviner propose les quelques mots suivants. Le modèle les
  vérifie tous d'un coup. Vous obtenez la même réponse, 1.1-1.2x plus tôt.
- **Les longs textes sont lus par gros morceaux** (1,024 tokens à la fois, les experts que la carte ne contient pas
  passant par elle), à 780 à 1,250 tokens par seconde.

Chaque partie et ses chiffres : [docs/GLM.md](GLM.md#how-it-uses-the-card) (en anglais).

## DeepSeek V4 Flash

<p align="center"><img src="media/how-it-works-dsv4.svg" width="860" alt="Les 11,008 experts du modèle : les plus utilisés sur la carte graphique, le reste en RAM, les fichiers du modèle sur le SSD"></p>

- **Le modèle est une équipe de 11,008 petits experts**, 256 dans chacune de ses 43 couches. Chaque mot n'en demande
  que 6 par couche, plus un que tous les mots utilisent.
- **Votre carte graphique** garde les plus utilisés, environ 2,840 sur une carte de 32 Go : le moteur compte ceux
  dont votre travail se sert. **Votre RAM** garde le reste, et **votre processeur** les calcule en même temps. **Votre
  SSD** garde les fichiers du modèle, utilisés là où ils sont, sans deuxième copie.
- **Une petite mémoire de la discussion :** un mot lit les 128 derniers mots en entier et le reste sous forme de
  lignes compressées, une pour 4 ou pour 128 mots. Un token de contexte prend 6,880 octets, si bien qu'un long contexte
  laisse à la carte presque toute sa place pour les experts.
- **Les longs textes sont lus en gros morceaux** (1,024 tokens à la fois), 30,000 tokens à environ 1,000 tokens par
  seconde. Il n'a pas encore de « deviner puis vérifier » : le modèle brouillon fait pour lui ne tourne pas encore.

Chaque partie et ses chiffres : [docs/DEEPSEEK.md](DEEPSEEK.md#how-it-differs-from-the-other-models) (en anglais).

## LFM2.5-24B-A2B

<p align="center"><img src="media/how-it-works-lfm25.svg" width="860" alt="Les 2,432 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Le modèle est une équipe de 2,432 petits experts**, 64 dans chacune de 38 couches. Chaque mot n'en demande que 4
  par couche.
- **La plupart de ses 40 couches n'ont pas d'attention.** 30 d'entre elles mélangent les mots par une courte
  convolution sur les trois derniers et ne gardent que deux vecteurs par couche. Seules les 10 autres gardent toute la
  discussion, si bien que sa mémoire d'une discussion est petite : un contexte de 128K ne prend que 2 Go de VRAM de
  plus que 32K.
- **Votre carte graphique** garde autant d'experts qu'elle peut : environ 900 à 1,300 sur une carte de 8 Go, tous à
  partir de 24 Go pour les tailles à 4 bits. Sur la carte de 8 Go, ceux qu'elle contient servent les deux tiers aux
  quatre cinquièmes de ce qu'une réponse utilise. **Votre RAM** contient les autres, et **votre processeur** les
  calcule en même temps. **Votre SSD** contient le fichier du modèle, utilisé là où il se trouve, sans seconde copie.
- **Pas de bloc de brouillon.** Le modèle n'a rien qui devine la suite, donc le moteur ne devine qu'à partir du texte
  que la discussion répète.
- **Quand la carte contient tous les experts**, toute une étape reste sur la carte : elle choisit elle-même les
  experts des mots et fait tourner les 40 couches d'un token comme un seul lancement enregistré.

Chaque partie et ses chiffres : [docs/LFM.md](LFM.md) (en anglais).

## Flint 40B-A2.5B

<p align="center"><img src="media/how-it-works-flint.svg" width="860" alt="Les 12,032 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Le modèle est une équipe de 12,032 petits experts**, 256 dans chacune de 47 couches. Chaque mot n'en demande que 6
  par couche, plus un expert partagé que chaque mot utilise.
- **La plupart de ses 49 couches regardent 513 tokens en arrière.** 35 d'entre elles ne voient que les 513 derniers
  tokens et gardent un anneau de lignes qui ne grandit pas. Seules les 14 autres gardent toute la discussion, si bien
  que sa mémoire d'une discussion est petite : un contexte de 128K ne prend que 2,6 Go de VRAM de plus que 32K.
- **Votre carte graphique** garde autant d'experts qu'elle peut : environ 1,600 à 2,500 sur une carte de 8 Go, tous
  ceux d'UD-Q4_K_XL sur une carte de 32 Go. **Votre RAM** contient les autres, et **votre processeur** les calcule en
  même temps. **Votre SSD** contient le fichier du modèle, utilisé là où il se trouve, sans seconde copie.
- **Pas de bloc de brouillon.** Le modèle n'a rien qui devine la suite, donc le moteur ne devine qu'à partir du texte
  que la discussion répète.

Chaque partie et ses chiffres : [docs/FLINT.md](FLINT.md) (en anglais).

## Qwen3.6-35B-A3B

<p align="center"><img src="media/how-it-works-qwen36.svg" width="860" alt="Les 10,240 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Le modèle est une équipe de 10,240 petits experts**, 256 dans chacune de 40 couches. Chaque mot n'en demande que 8
  par couche.
- **Votre carte graphique** en garde autant qu'elle peut : environ 2,600 sur une carte de 8 Go, tous à partir de 24
  Go. Pendant qu'il écrit, le moteur fait venir sur la carte les experts que la réponse choisit sans cesse, et sur la
  carte de 8 Go ceux qu'elle contient servent environ 60 % de ce qu'une réponse utilise. **Votre RAM** contient les
  autres, et **votre processeur** les calcule en même temps. **Votre SSD** contient le fichier du modèle, utilisé là
  où il se trouve, sans seconde copie.

<p align="center"><img src="media/guess-and-check-qwen36.svg" width="860" alt="Le bloc brouillon du modèle devine les mots suivants ; le modèle les vérifie tous d'un coup et garde les bons"></p>

- **Deviner, puis vérifier :** le bloc brouillon du modèle lui-même, qui est dans son fichier, propose les quelques
  mots suivants. Le modèle les vérifie tous d'un coup. Vous obtenez la même réponse, 1.2-1.5x plus tôt.
- **Les longs textes sont lus par gros morceaux** (1,024 tokens à la fois). Quand la carte ne contient pas tous les
  experts, les autres passent par elle une seule fois pour tout le prompt.

Chaque partie et ses chiffres : [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en anglais).

## Clover-35B-A3B-Preview

<p align="center"><img src="media/how-it-works-clover.svg" width="860" alt="Les 10,240 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Qwen3.6-35B-A3B entraîné plus loin :** une équipe de 10,240 petits experts, 256 dans chacune de 40 couches, et
  seulement 8 par couche pour chaque mot.
- **Votre carte graphique** en garde autant qu'elle peut : environ 2,200 à 4,100 sur une carte de 8 Go, d'autant plus
  que la taille est petite, et tous ceux de Q3_K_M à partir de 24 Go. Sur la carte de 8 Go, ceux qu'elle contient
  servent 60 à 80 % de ce qu'une réponse utilise. **Votre RAM** contient les autres, et **votre processeur** les calcule
  en même temps. **Votre SSD** contient le fichier du modèle, utilisé là où il se trouve, sans seconde copie.
- **Pas de bloc de brouillon.** Clover a été entraîné à écrire un bloc de mots d'un coup en remplissant des blancs
  ensemble, et il peut ainsi deviner les mots suivants. Mais chaque bloc deviné refait tourner tout le modèle, ce qui a
  coûté plus que les bonnes devinettes n'ont fait gagner, donc c'est désactivé
  ([mesures](BACKENDS.md#clover-35b-a3b-preview), en anglais). Le moteur ne devine qu'à partir du texte que la
  discussion répète.
- **Les longs textes sont lus par gros morceaux** (1,024 tokens à la fois), comme avec Qwen3.6-35B-A3B.

Chaque partie et ses chiffres : [docs/BACKENDS.md](BACKENDS.md#clover-35b-a3b-preview) (en anglais).

## Ornith 1.5

<p align="center"><img src="media/how-it-works-ornith.svg" width="860" alt="Les 10,240 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **L'architecture de Qwen3.6-35B-A3B avec d'autres poids :** une équipe de 10,240 petits experts, 256 dans chacune de
  40 couches, et seulement 8 par couche pour chaque mot.
- **Votre carte graphique** en garde autant qu'elle peut : environ 3,100 sur une carte de 8 Go, tous à partir de 24
  Go. Pendant qu'il écrit, le moteur fait venir sur la carte les experts que la réponse choisit sans cesse, et sur la
  carte de 8 Go ceux qu'elle contient servent environ 60 % de ce qu'une réponse utilise. **Votre RAM** contient les
  autres, et **votre processeur** les calcule en même temps. **Votre SSD** contient le fichier du modèle, utilisé là
  où il se trouve, sans seconde copie.

<p align="center"><img src="media/guess-and-check-ornith.svg" width="860" alt="Le bloc brouillon du modèle devine les mots suivants ; le modèle les vérifie tous d'un coup et garde les bons"></p>

- **Deviner, puis vérifier :** son bloc brouillon est aussi dans son fichier et propose les quelques mots suivants. Il
  devine juste moins souvent que celui de Qwen3.6-35B-A3B : vous obtenez la même réponse 1.1-1.2x plus tôt.
- **Les longs textes sont lus par gros morceaux** (1,024 tokens à la fois). Quand la carte ne contient pas tous les
  experts, les autres passent par elle une seule fois pour tout le prompt.

Chaque partie et ses chiffres : [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en anglais).

## Gemma 4 26B-A4B

<p align="center"><img src="media/how-it-works-gemma4.svg" width="860" alt="Les 3,840 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Le modèle est une équipe de 3,840 petits experts**, 128 dans chacune de 30 couches. Chaque mot n'en demande que 8
  par couche, à côté d'une partie de la couche que tous les mots utilisent.
- **Votre carte graphique** en garde autant qu'elle peut : environ 1,000 sur une carte de 8 Go, tous à partir de 16 Go
  d'après l'estimation de l'installeur. Pendant qu'il écrit, le moteur fait venir sur la carte les experts que la
  réponse choisit sans cesse, et sur la carte de 8 Go ceux qu'elle contient servent environ 55 % de ce qu'une réponse
  utilise. **Votre RAM** contient les autres, et **votre processeur** les calcule en même temps. **Votre SSD**
  contient le fichier du modèle et son modèle brouillon, utilisés là où ils se trouvent, sans seconde copie.

<p align="center"><img src="media/guess-and-check-gemma4.svg" width="860" alt="Le petit modèle brouillon de Google devine les mots suivants ; le modèle les vérifie tous d'un coup et garde les bons"></p>

- **Deviner, puis vérifier :** le petit modèle brouillon de Google pour lui, un fichier de 0.5 Go que l'installeur
  télécharge avec le modèle, propose les quelques mots suivants. Le modèle les vérifie tous d'un coup. Vous obtenez la
  même réponse, 1.3-1.6x plus tôt.
- **Les longs textes sont lus par gros morceaux** (1,024 tokens à la fois). Quand la carte ne contient pas tous les
  experts, les autres passent par elle une seule fois pour tout le prompt.

Chaque partie et ses chiffres : [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en anglais).

## Tiel Coder 35B-A3B

<p align="center"><img src="media/how-it-works-tiel.svg" width="860" alt="Les 10,240 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Ornith 1.5 dans une autre quantification :** une équipe de 10,240 petits experts, 256 dans chacune de
  40 couches, et seulement 8 par couche pour chaque mot.
- **Votre carte graphique** en garde autant qu'elle peut : environ 2,700 sur une carte de 8 Go, tous à partir de 24
  Go. Pendant qu'il écrit, le moteur fait venir sur la carte les experts que la réponse choisit sans cesse, et sur la
  carte de 8 Go ceux qu'elle contient servent environ 50 % de ce qu'une réponse utilise. **Votre RAM** contient les
  autres, et **votre processeur** les calcule en même temps. **Votre SSD** contient le fichier du modèle, utilisé là
  où il se trouve, sans seconde copie.

<p align="center"><img src="media/guess-and-check-tiel.svg" width="860" alt="Le bloc brouillon du modèle devine les mots suivants ; le modèle les vérifie tous d'un coup et garde les bons"></p>

- **Deviner, puis vérifier :** son bloc brouillon est aussi dans son fichier et propose les quelques mots suivants. Il
  devine juste moins souvent que celui de Qwen3.6-35B-A3B : vous obtenez la même réponse 1.2-1.3x plus tôt.
- **Les longs textes sont lus par gros morceaux** (1,024 tokens à la fois). Quand la carte ne contient pas tous les
  experts, les autres passent par elle une seule fois pour tout le prompt.

Chaque partie et ses chiffres : [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en anglais).

## MiniCPM-V 4.7 35B-A3B

<p align="center"><img src="media/how-it-works-minicpmv.svg" width="860" alt="Les 10,240 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **L'architecture de Qwen3.6-35B-A3B avec les poids d'OpenBMB :** une équipe de 10,240 petits experts, 256 dans
  chacune de 40 couches, et seulement 8 par couche pour chaque mot.
- **Votre carte graphique** en garde autant qu'elle peut : avec Q4_K_M environ 2,000 sur une carte de 8 Go, tous sur
  une carte de 32 Go. Pendant qu'il écrit, le moteur fait venir sur la carte les experts que la réponse choisit sans
  cesse, et sur la carte de 8 Go ceux qu'elle contient servent environ 60 à 70 % de ce qu'une réponse utilise.
  **Votre RAM** contient les autres, et **votre processeur** les calcule en même temps. **Votre SSD** contient le
  fichier du modèle, utilisé là où il se trouve, sans seconde copie.
- **Un mot à la fois :** son fichier n'a pas de bloc brouillon, donc rien ne devine à l'avance.
- **Images :** son propre encodeur d'images lit une grande image comme une vue d'ensemble et quelques tuiles. Une
  capture de 1,200 sur 398 pixels devient environ 300 tokens, un graphique de 2,520 sur 1,620 environ 600.
- **Les longs textes sont lus par gros morceaux** (1,024 tokens à la fois). Quand la carte ne contient pas tous les
  experts, les autres passent par elle une seule fois pour tout le prompt.

Chaque partie et ses chiffres : [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en anglais).

## Kolibri 1

<p align="center"><img src="media/how-it-works-kolibri.svg" width="860" alt="Les 19,200 experts du modèle : ceux qu'une réponse choisit sans cesse sur la carte graphique, les autres en RAM, le fichier du modèle sur le SSD"></p>

- **Le modèle est une équipe de 19,200 petits experts**, 384 dans chacune de 50 couches. Chaque mot n'en demande que 6
  par couche, plus un expert partagé que chaque mot utilise.
- **Quatre couches sur cinq regardent 513 tokens en arrière.** Elles gardent un anneau de lignes qui ne grandit pas.
  La cinquième garde toute la discussion et n'utilise aucune position des mots.
- **Votre carte graphique** garde autant d'experts qu'elle peut : environ un dixième sur une carte de 8 Go, tous ceux de
  Q2_K sur une carte de 32 Go. **Votre RAM** contient les autres, et **votre processeur** les calcule en même temps.
  **Votre SSD** contient le fichier du modèle, utilisé là où il se trouve, sans seconde copie.
- **Pas de bloc de brouillon.** Le modèle n'a rien qui devine la suite, donc le moteur ne devine qu'à partir du texte
  que la discussion répète.

Chaque partie et ses chiffres : [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en anglais).

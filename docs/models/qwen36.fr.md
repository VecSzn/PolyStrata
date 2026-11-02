# Qwen3.6-35B-A3B

Retour à [README.fr.md](../../README.fr.md)

**Qwen3.6-35B-A3B**, un fichier de 13 à 23 Go. Il lit aussi les images.

## Quelle est sa vitesse ?

Mesuré par PolyStrata sur deux PC, avec une courte discussion et un prompt de 30,000 à 32,000 tokens : à gauche un
portable dont la carte a 8 Go, où le processeur calcule les experts que la carte ne peut pas contenir, à droite une
station de travail dont la carte contient le modèle entier.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 94-100 tokens/s | 1,690 tokens/s |
| **UD-Q2_K_XL** | 123-124 tokens/s | 1,590 tokens/s |
| **UD-Q4_K_M** | 82-86 tokens/s | 1,560-1,570 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 394-395 tokens/s | 7,800-7,810 tokens/s |
| **UD-Q2_K_XL** | 450 tokens/s | 7,720-7,740 tokens/s |
| **UD-Q4_K_M** | 395-396 tokens/s | 8,070-8,080 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens est lu à 1,660 tokens/s sur le portable et à 6,300 sur la station de travail.
Les chiffres de la carte de la
station limitée à 8 Go, un substitut d'avant la mesure sur le portable, sont dans
[docs/BACKENDS.md](../BACKENDS.md#with-8-gb-of-vram) (en anglais).

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 5 Go sont pris avant le premier expert. Il a tourné sur une carte de 32 Go, sur cette carte limitée à 8 Go, et sur la carte de 8 Go d'un portable. Sur une carte AMD, il tourne aussi, en bêta, sous Linux et sous Windows : l'installateur compile le moteur sur votre PC ; testé pour l'instant avec Qwen3.6 et Ornith sur une RX 7600. |
| **RAM** | Environ **17, 21 ou 27 Go** à côté d'une carte de 8 Go pour les trois tailles, d'après l'estimation de l'installeur. Une carte de 24 Go contient en entier les deux plus petites. Les trois tailles ont tourné sur un portable de 32 Go. |
| **Disque** | 13, 17 ou 23 Go pour le fichier du modèle, 1 Go de plus pour lire les images. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour GLM 5.3 Flash. Sous Windows, il télécharge le moteur déjà compilé
de la dernière version publiée. Sur un processeur sans AVX2, ou dans une copie dont le moteur a changé depuis cette
version, il le compile à la place avec le compilateur 2022 de Visual Studio et le CUDA Toolkit, et propose de les
installer. Pas encore : deux cartes sous Windows, trois cartes ou plus. Sur une carte AMD sous Windows, l'installeur
compile `strata-poly` pour la carte comme sous Linux (les wheels TheRock d'AMD dans `.rocm-win`) et prévient que l'y
faire tourner est nouveau ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

Depuis la 0.1.44, le moteur de [Strata](https://github.com/Niko1221/Strata) lui-même fait aussi tourner Qwen3.6, à partir de
UD-Q4_K_XL (`--family qwen36-strata`, 32 Go de RAM ou plus). Sur une RTX 5090 avec UD-Q4_K_XL, il a lu un prompt de 4K
1,6 fois plus vite que strata-poly ; sur un prompt de 31K et en écriture, les deux sont à peu près à égalité
([mesures](../BACKENDS.md#qwen36-on-two-engines), en anglais). Les tailles plus petites, pour les cartes de 8 Go, restent sur strata-poly.

## Quel modèle choisir ?

Trois tailles pour l'instant. L'installeur prend UD-Q3_K_M, ou UD-Q2_K_XL sur un PC qui a moins de RAM. UD-Q4_K_M se choisit par son nom : `--model UD-Q4_K_M`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | le modèle entier avec des experts à 3 bits ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 Go | environ 21 Go |
| **UD-Q2_K_XL** | le même avec des experts à 2 et 3 bits : celui qui demande le moins de RAM | 13 Go | environ 17 Go |
| **UD-Q4_K_M** | le même avec des experts à 4 bits | 23 Go | environ 27 Go |

Une carte plus grande contient une plus grande part du modèle et demande moins de RAM : avec UD-Q3_K_M, environ 17 Go à côté d'une carte de 12 Go, 13 Go à côté d'une carte de 16 Go, et 10 Go dès que la carte contient tout. Ce sont les estimations
de l'installeur. Elles n'ont pas été vérifiées sur un tel PC.

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#qwen36-35b-a3b).

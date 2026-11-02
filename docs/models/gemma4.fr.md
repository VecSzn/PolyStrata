# Gemma 4 26B-A4B

Retour à [README.fr.md](../../README.fr.md)

**Gemma 4 26B-A4B**, un fichier de 11 à 17 Go. Il lit aussi les images.

## Quelle est sa vitesse ?

Mesuré comme [Qwen3.6-35B-A3B](qwen36.fr.md), sur les deux mêmes PC.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 92-93 tokens/s | 2,240 tokens/s |
| **UD-Q2_K_XL** | 101-104 tokens/s | 2,150-2,160 tokens/s |
| **UD-Q4_K_M** | 85 tokens/s | 2,150 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 353-356 tokens/s | 8,420-8,480 tokens/s |
| **UD-Q2_K_XL** | 332-338 tokens/s | 7,940-7,970 tokens/s |
| **UD-Q4_K_M** | 335-346 tokens/s | 8,320-8,390 tokens/s |
| **QAT-UD-Q4_K_XL** | 356-371 tokens/s | 8,460-8,680 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens est lu à 2,130 tokens/s sur le portable et à 6,800 sur la station de travail. QAT-UD-Q4_K_XL a été mesuré plus tard, avec le moteur de v0.1.45-poly.4 ; UD-Q4_K_M, mesuré à côté avec ce moteur, a écrit 333-353 tokens/s et lu 8,510-8,570. Sur le portable, avec d'autres programmes ouverts et peu de RAM, il a tourné en alternance avec UD-Q4_K_M : il a écrit 50-62 tokens/s dans un chat contre 47-48, lu le long prompt à 1,680-1,730 contre 1,660-1,670 et écrit ensuite 65-68 contre 48-49.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 6 Go sont pris avant le premier expert. Il a tourné sur une carte de 32 Go, sur cette carte limitée à 8 Go, et sur la carte de 8 Go d'un portable. Sur une carte AMD, il tourne aussi, en bêta, sous Linux et sous Windows : l'installateur compile le moteur sur votre PC ; des utilisateurs l'ont fait tourner sur une RX 6700 XT (12 Go) sous Linux. |
| **RAM** | Environ **15, 18, 19 ou 21 Go** à côté d'une carte de 8 Go pour les quatre tailles, d'après l'estimation de l'installeur. D'après la même estimation, une carte de 16 Go contient en entier les deux plus petites et le fichier QAT. Les quatre tailles ont tourné sur un portable de 32 Go. |
| **Disque** | 11, 13, 15 ou 17 Go pour le fichier du modèle et son modèle brouillon, 1 Go de plus pour lire les images. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour GLM 5.3 Flash. Sous Windows, il télécharge le moteur déjà compilé
de la dernière version publiée. Sur un processeur sans AVX2, ou dans une copie dont le moteur a changé depuis cette
version, il le compile à la place avec le compilateur 2022 de Visual Studio et le CUDA Toolkit, et propose de les
installer. Pas encore : deux cartes sous Windows, trois cartes ou plus. Sur une carte AMD sous Windows, l'installeur
compile `strata-poly` pour la carte comme sous Linux (les wheels TheRock d'AMD dans `.rocm-win`) et prévient que l'y
faire tourner est nouveau ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Quel modèle choisir ?

Quatre tailles pour l'instant. L'installeur prend UD-Q3_K_M, ou UD-Q2_K_XL sur un PC qui a moins de RAM. Les deux autres se choisissent par leur nom : `--model UD-Q4_K_M`, `--model QAT-UD-Q4_K_XL`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | le modèle entier avec des experts à 3 bits ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)), et le modèle brouillon de Google pour lui | 13 Go | environ 18 Go |
| **UD-Q2_K_XL** | le même avec des experts à 2 bits : celui qui demande le moins de RAM | 11 Go | environ 15 Go |
| **UD-Q4_K_M** | le même avec des experts à 4 et 5 bits | 17 Go | environ 21 Go |
| **QAT-UD-Q4_K_XL** | les poids de Google entraînés avec la quantification (QAT), en 4 bits ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)), avec leur propre modèle brouillon | 15 Go | environ 19 Go |

Une carte plus grande contient une plus grande part du modèle et demande moins de RAM : avec UD-Q3_K_M, environ 14 Go à côté d'une carte de 12 Go, et 10 Go dès que la carte contient tout. Ce sont les estimations
de l'installeur. Elles n'ont pas été vérifiées sur un tel PC.

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#gemma-4-26b-a4b).

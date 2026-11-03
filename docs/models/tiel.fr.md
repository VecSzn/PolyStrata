# Tiel Coder 35B-A3B

Retour à [README.fr.md](../../README.fr.md)

**Tiel Coder 35B-A3B**, Ornith 1.5 quantifié par peculiar-ragdoll pour les agents de code : un fichier de 14 à 23 Go. Il lit aussi les images.

## Quelle est sa vitesse ?

Mesuré comme [Qwen3.6-35B-A3B](qwen36.fr.md), sur les deux mêmes PC.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 83 tokens/s | 1,700 tokens/s |
| **UD-IQ3_XXS** | 96-97 tokens/s | 1,610 tokens/s |
| **UD-Q4_K_XL** | 74-79 tokens/s | 1,660-1,670 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 337-345 tokens/s | 8,080-8,110 tokens/s |
| **UD-IQ3_XXS** | 352 tokens/s | 7,520 tokens/s |
| **UD-Q4_K_XL** | 316-322 tokens/s | 7,750-7,970 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens est lu à 1,670 tokens/s sur le portable et à 6,300 sur la station de travail.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 5 Go sont pris avant le premier expert. Il a tourné sur une carte de 32 Go, sur cette carte limitée à 8 Go, et sur la carte de 8 Go d'un portable. Sur une carte AMD, il tourne aussi, en bêta, sous Linux et sous Windows : l'installateur compile le moteur sur votre PC ; des utilisateurs l'ont fait tourner sur une RX 6700 XT et une RX 7800 XT sous Linux, et sur une RX 9070 XT sous Windows. |
| **RAM** | Environ **18, 21 ou 27 Go** à côté d'une carte de 8 Go pour les trois tailles, d'après l'estimation de l'installeur. Une carte de 24 Go contient en entier les trois tailles. Les trois tailles ont tourné sur un portable de 32 Go. |
| **Disque** | 14, 17 ou 23 Go pour le fichier du modèle, 1 Go de plus pour lire les images. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour GLM 5.3 Flash. Sous Windows, il télécharge le moteur déjà compilé
de la dernière version publiée. Sur un processeur sans AVX2, ou dans une copie dont le moteur a changé depuis cette
version, il le compile à la place avec le compilateur 2022 de Visual Studio et le CUDA Toolkit, et propose de les
installer. Pas encore : deux cartes sous Windows, trois cartes ou plus. Sur une carte AMD sous Windows, l'installeur
compile `strata-poly` pour la carte comme sous Linux (les wheels TheRock d'AMD dans `.rocm-win`) et prévient que l'y
faire tourner est nouveau ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Quel modèle choisir ?

Trois tailles pour l'instant. L'installeur prend UD-Q3_K_XL, ou UD-IQ3_XXS sur un PC qui a moins de RAM. UD-Q4_K_XL se choisit par son nom : `--model UD-Q4_K_XL`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **UD-Q3_K_XL** | le modèle entier avec des experts à 3 et 4 bits ([peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)) | 17 Go | environ 21 Go |
| **UD-IQ3_XXS** | le même avec des experts à 2 et 3 bits : celui qui demande le moins de RAM | 14 Go | environ 18 Go |
| **UD-Q4_K_XL** | le même avec des experts de 4 à 6 bits | 23 Go | environ 27 Go |

Une carte plus grande contient une plus grande part du modèle et demande moins de RAM : avec UD-Q3_K_XL, environ 17 Go à côté d'une carte de 12 Go, 13 Go à côté d'une carte de 16 Go, et 10 Go dès que la carte contient tout. Ce sont les estimations
de l'installeur. Elles n'ont pas été vérifiées sur un tel PC.

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#tiel-coder-35b-a3b).

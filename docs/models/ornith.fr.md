# Ornith 1.5

Retour à [README.fr.md](../../README.fr.md)

**Ornith 1.5**, qui a l'architecture de Qwen3.6-35B-A3B et d'autres poids : un fichier de 14 à 24 Go. Il lit aussi les images.

## Quelle est sa vitesse ?

Mesuré comme [Qwen3.6-35B-A3B](qwen36.fr.md), sur les deux mêmes PC.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 76-82 tokens/s | 1,620 tokens/s |
| **APEX-I-MINI** | 80-86 tokens/s | 1,550-1,560 tokens/s |
| **APEX-I-QUALITY** | 59-63 tokens/s | 1,570-1,580 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 316-319 tokens/s | 8,040-8,110 tokens/s |
| **APEX-I-MINI** | 289-300 tokens/s | 7,710-7,740 tokens/s |
| **APEX-I-QUALITY** | 297-302 tokens/s | 7,860-7,880 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens est lu à 1,590 tokens/s sur le portable et à 6,400 sur la station de travail.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 4 Go sont pris avant le premier expert. Il a tourné sur une carte de 32 Go, sur cette carte limitée à 8 Go, et sur la carte de 8 Go d'un portable. Sur une carte AMD, il tourne aussi, en bêta, sous Linux et sous Windows : l'installateur compile le moteur sur votre PC ; testé pour l'instant avec Qwen3.6 et Ornith sur une RX 7600. |
| **RAM** | Environ **18, 22 ou 28 Go** à côté d'une carte de 8 Go pour les trois tailles, d'après l'estimation de l'installeur. Une carte de 24 Go contient en entier les deux plus petites. Les trois tailles ont tourné sur un portable de 32 Go. |
| **Disque** | 14, 17 ou 24 Go pour le fichier du modèle, 1 Go de plus pour lire les images. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour GLM 5.3 Flash. Sous Windows, il télécharge le moteur déjà compilé
de la dernière version publiée. Sur un processeur sans AVX2, ou dans une copie dont le moteur a changé depuis cette
version, il le compile à la place avec le compilateur 2022 de Visual Studio et le CUDA Toolkit, et propose de les
installer. Pas encore : deux cartes sous Windows, trois cartes ou plus. Sur une carte AMD sous Windows, l'installeur
compile `strata-poly` pour la carte comme sous Linux (les wheels TheRock d'AMD dans `.rocm-win`) et prévient que l'y
faire tourner est nouveau ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Quel modèle choisir ?

Trois tailles pour l'instant. L'installeur prend APEX-I-COMPACT, ou APEX-I-MINI sur un PC qui a moins de RAM. APEX-I-QUALITY se choisit par son nom : `--model APEX-I-QUALITY`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | le modèle entier avec des experts à 3 et 4 bits ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) de mudler) | 17 Go | environ 22 Go |
| **APEX-I-MINI** | le même avec des experts à 2 et 3 bits : celui qui demande le moins de RAM | 14 Go | environ 18 Go |
| **APEX-I-QUALITY** | le même avec des experts de 4 à 6 bits | 24 Go | environ 28 Go |

Une carte plus grande contient une plus grande part du modèle et demande moins de RAM : avec APEX-I-COMPACT, environ 17 Go à côté d'une carte de 12 Go, 13 Go à côté d'une carte de 16 Go, et 10 Go dès que la carte contient tout. Ce sont les estimations
de l'installeur. Elles n'ont pas été vérifiées sur un tel PC.

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#ornith-15).

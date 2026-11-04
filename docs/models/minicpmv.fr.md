# MiniCPM-V 4.7 35B-A3B

Retour à [README.fr.md](../../README.fr.md)

**MiniCPM-V 4.7 35B-A3B** d'OpenBMB, avec l'architecture de Qwen3.6-35B-A3B et ses propres poids, fait pour lire les images : un fichier de 16 à 29 Go.

## Quelle est sa vitesse ?

Mesuré comme [Qwen3.6-35B-A3B](qwen36.fr.md), sur les deux mêmes PC. Son fichier n'a pas de bloc brouillon : il écrit un token à la fois.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 64 tokens/s | 1,150 tokens/s |
| **mix_q3gud_im** | 59 tokens/s | 1,130-1,150 tokens/s |
| **Q6_K** | 40 tokens/s | 960-980 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 308-311 tokens/s | 7,910-7,950 tokens/s |
| **mix_q3gud_im** | 291-293 tokens/s | 7,730-7,740 tokens/s |
| **Q6_K** | 277-278 tokens/s | 7,290-7,330 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens est lu à 1,260 tokens/s sur le portable et à 8,790 sur la station de travail. Sur le
portable, Q6_K remplit les 32 Go de RAM, c'est ce qui le ralentit là.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 3.5 Go sont pris avant le premier expert. Il a tourné sur une carte de 32 Go et sur la carte de 8 Go d'un portable. Sur une carte AMD, il tourne aussi, en bêta, sous Linux et sous Windows : l'installateur compile le moteur sur votre PC ; il a tourné sur une RX 7600. |
| **RAM** | Environ **19, 25 ou 31 Go** à côté d'une carte de 8 Go pour les trois tailles, d'après l'estimation de l'installeur. Une carte de 32 Go contient en entier les trois tailles. Les trois tailles ont tourné sur un portable de 32 Go. |
| **Disque** | 16, 21 ou 29 Go pour le fichier du modèle, 1 Go de plus pour lire les images. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour GLM 5.3 Flash. Sous Windows, il télécharge le moteur déjà compilé
de la dernière version publiée. Sur un processeur sans AVX2, ou dans une copie dont le moteur a changé depuis cette
version, il le compile à la place avec le compilateur 2022 de Visual Studio et le CUDA Toolkit, et propose de les
installer. Pas encore : deux cartes sous Windows, trois cartes ou plus. Sur une carte AMD sous Windows, l'installeur
compile `strata-poly` pour la carte comme sous Linux (les wheels TheRock d'AMD dans `.rocm-win`) et prévient que l'y
faire tourner est nouveau ([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Quel modèle choisir ?

Trois tailles pour l'instant, d'OpenBMB. L'installeur prend Q4_K_M, ou mix_q3gud_im sur un PC qui a moins de RAM. Q6_K se choisit par son nom : `--model Q6_K`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **Q4_K_M** | le modèle entier avec des experts à 4 bits ([OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)) | 21 Go | environ 25 Go |
| **mix_q3gud_im** | le même avec des experts à 3 et 4 bits : celui qui demande le moins de RAM | 16 Go | environ 19 Go |
| **Q6_K** | le même avec des experts à 6 bits | 29 Go | environ 31 Go |

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#minicpm-v-47-35b-a3b).

# LFM2.5-24B-A2B

Retour à [README.fr.md](../../README.fr.md)

**LFM2.5-24B-A2B** de Liquid AI, un fichier de 11 à 25 Go. La plupart de ses couches n'ont pas d'attention, si bien qu'une longue discussion prend peu de VRAM : un contexte de 128K ne prend que 2 Go de plus que 32K. Sa licence n'autorise l'usage commercial qu'aux entreprises de moins de 10 millions de dollars de chiffre d'affaires annuel.

## Quelle est sa vitesse ?

Mesuré comme [Qwen3.6-35B-A3B](qwen36.fr.md), sur les deux mêmes PC, avec un prompt d'environ 30,000 tokens.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 70-73 tokens/s | 3,590-3,620 tokens/s |
| **UD-Q3_K_XL** | 88 tokens/s | 3,760-3,810 tokens/s |
| **QAD-Q4_0** | 77-78 tokens/s | 3,720-3,730 tokens/s |
| **UD-Q6_K_XL** | 42-43 tokens/s | 3,030-3,120 tokens/s |
| **Q8_0** | 23-34 tokens/s | 2,950-3,040 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 480-485 tokens/s | 23,390-23,460 tokens/s |
| **UD-Q3_K_XL** | 490-497 tokens/s | 22,920-23,400 tokens/s |
| **QAD-Q4_0** | 498 tokens/s | 23,880-23,900 tokens/s |
| **UD-Q6_K_XL** | 423-426 tokens/s | 21,250-21,270 tokens/s |
| **Q8_0** | 376 tokens/s | 23,270-23,350 tokens/s |

</td></tr>
</table>

Sur le portable, Q8_0 remplit les 32 Go de RAM, d'où sa lenteur ; la station de travail contient chaque taille en
entier.

Sur le portable, llama.cpp b11540 avec `--n-cpu-moe 25`, le meilleur des réglages essayés là, a lu un prompt de 32,000
tokens à 530 tokens/s et a écrit ensuite à 53 ; PolyStrata l'a lu à 3,540 et a écrit à 71-73. Sur la station de
travail, llama-bench a lu 29,816 tokens à 13,300 tokens/s avec le lot par défaut de llama.cpp, 512, et à 23,600 avec
2,048, aussi vite que PolyStrata, et a écrit token par token à 455 tokens/s, et à 369 après ce prompt ; PolyStrata
écrit à 452 et 387. Un prompt d'environ 124,000 tokens dans un contexte de 128K a été lu et a reçu sa réponse en 8,7
secondes sur la station de travail ; le portable l'a lu en 45 secondes.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 4 Go sont pris avant le premier expert. Il a tourné sur la carte de 8 Go d'un portable et sur une carte de 32 Go. |
| **RAM** | Environ **16, 18, 20, 27 ou 31 Go** à côté d'une carte de 8 Go pour UD-Q3_K_XL, QAD-Q4_0, UD-Q4_K_XL, UD-Q6_K_XL et Q8_0, d'après l'estimation de l'installeur. Une carte de 24 Go contient en entier les trois plus petites. Les cinq tailles ont tourné sur un portable de 32 Go. |
| **Disque** | 11 à 25 Go pour le fichier du modèle. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour [Qwen3.6-35B-A3B](qwen36.fr.md). Un utilisateur l'a fait tourner sur une
AMD RX 6700 XT (12 Go) sous Linux. Il ne lit que du texte. La licence de Liquid AI n'autorise l'usage commercial qu'aux entreprises de moins
de 10 millions de dollars de chiffre d'affaires annuel ([LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)).

La liste complète : [docs/LFM.md](../LFM.md#what-you-need) (en anglais).

## Quel modèle choisir ?

Cinq tailles. L'installeur prend UD-Q4_K_XL, ou UD-Q3_K_XL sur un PC qui a moins de RAM. Les autres se choisissent par leur nom : `--model QAD-Q4_0`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | le modèle entier avec des experts à 4 bits ([Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)) | 15 Go | environ 20 Go |
| **UD-Q3_K_XL** | le même avec des experts à 3 bits : celui qui demande le moins de RAM | 11 Go | environ 16 Go |
| **QAD-Q4_0** | 4 bits, de [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) même, entraîné pour perdre moins à cette quantification | 14 Go | environ 18 Go |
| **UD-Q6_K_XL** | des experts à 6 bits | 22 Go | environ 27 Go |
| **Q8_0** | 8 bits partout, le plus proche du modèle tel qu'entraîné | 25 Go | environ 31 Go |

Comparées à Q8_0, sur cinq prompts de 256 tokens chacun, les autres tailles choisissent le token le plus probable de Q8_0
à 96,8 % des pas (UD-Q6_K_XL), 95,5 % (UD-Q4_K_XL), 92,7 % (QAD-Q4_0) et 91,6 % (UD-Q3_K_XL).

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#lfm25-24b-a2b).

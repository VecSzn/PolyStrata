# Flint 40B-A2.5B

Retour à [README.fr.md](../../README.fr.md)

**Flint 40B-A2.5B** de Recursive, fait pour les agents de code : un fichier de 18 à 42 Go. La plupart de ses couches ne regardent que 513 tokens en arrière, si bien qu'une longue discussion prend peu de VRAM : un contexte de 128K ne prend que 2,6 Go de plus que 32K.

## Quelle est sa vitesse ?

Mesuré comme [LFM2.5-24B-A2B](lfm25.fr.md), sur les deux mêmes PC, avec un prompt d'environ 29,000 tokens.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 90 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 20-29 tokens/s | 1,310-1,420 tokens/s |
| **UD-Q3_K_XL** | 61-63 tokens/s | 2,000-2,010 tokens/s |
| **UD-Q6_K_XL** | Ne tient pas | Ne tient pas |
| **Q8_0** | Ne tient pas | Ne tient pas |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 251-252 tokens/s | 12,900 tokens/s |
| **UD-Q3_K_XL** | 260-264 tokens/s | 13,080-13,200 tokens/s |
| **Q8_0** | 158-160 tokens/s | 6,740-6,970 tokens/s |

</td></tr>
</table>

Sur le portable, seule UD-Q3_K_XL reste dans les 32 Go de RAM à côté des programmes du bureau. Les plus grandes
tailles relisent sans cesse leurs experts sur le SSD, d'où leur lenteur là, et l'installeur y prend UD-Q3_K_XL. La
carte de la station de travail contient UD-Q4_K_XL et UD-Q3_K_XL en entier ; UD-Q6_K_XL n'y a pas tourné.

Sur le portable, llama.cpp avec `--fit on`, le plus rapide des réglages essayés là, a lu un prompt de 29,442 tokens
à 390 tokens/s et a écrit ensuite à 35 ; PolyStrata l'a lu à 1,950-2,010 et a écrit à 49-52
([les mesures](../../bench/results/2026-10-24-laptop-flint-llamacpp-a37cc46/README.md)).

Sur la station de travail, llama-bench de llama.cpp a lu 29,816 tokens à 7,790 tokens/s avec son lot par défaut de
512 et à 12,470 avec 2,048, PolyStrata à 12,910-12,940 ; token par token, llama.cpp a écrit à 222 tokens/s et à 188 après
ce prompt, PolyStrata à 249-256 et 212-213. Un prompt de 124,122 tokens dans un contexte de 128K a été lu et a reçu sa
réponse en 20 secondes, et les trois faits cachés dedans ont été trouvés.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 4 Go sont pris avant le premier expert. Il a tourné sur la carte de 8 Go d'un portable et sur une carte de 32 Go. |
| **RAM** | Environ **22, 29, 40 ou 46 Go** à côté d'une carte de 8 Go pour UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL et Q8_0, d'après l'estimation de l'installeur. Sur un portable de 32 Go, seule UD-Q3_K_XL est restée en RAM, donc l'installeur ne prend UD-Q4_K_XL qu'à partir d'environ 40 Go. Une carte de 32 Go contient UD-Q4_K_XL en entier. |
| **Disque** | 18 à 42 Go pour le fichier du modèle. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour [Qwen3.6-35B-A3B](qwen36.fr.md). Il n'a pas encore été essayé sur une
carte AMD. Il ne lit que du texte. Sa licence est Apache 2.0.

La liste complète : [docs/FLINT.md](../FLINT.md#what-you-need) (en anglais).

## Quel modèle choisir ?

Quatre tailles. À côté d'une carte de 8 Go, l'installeur prend UD-Q4_K_XL à partir d'environ 40 Go de RAM et
UD-Q3_K_XL en dessous ; à côté d'une carte plus grande, UD-Q4_K_XL demande moins. Les autres se choisissent par leur
nom : `--model Q8_0`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | le modèle entier avec des experts à 4 et 5 bits ([Unsloth](https://huggingface.co/unsloth/flint-GGUF)) | 25 Go | environ 29 Go |
| **UD-Q3_K_XL** | le même avec des experts à 3 bits : le moins de RAM | 18 Go | environ 22 Go |
| **UD-Q6_K_XL** | experts à 6 bits | 36 Go | environ 40 Go |
| **Q8_0** | 8 bits partout, le plus proche du modèle tel qu'entraîné | 42 Go | environ 46 Go |

Sur le portable de 32 Go, le fichier d'UD-Q4_K_XL n'est pas resté en RAM : il a écrit une courte discussion à 20-29
tokens/s, UD-Q3_K_XL à 61-63. Comparées à Q8_0, sur cinq prompts de 256 tokens chacun, les autres tailles choisissent
le token le plus probable de Q8_0 à 97,8 % des pas (UD-Q6_K_XL), 95,9 % (UD-Q4_K_XL) et 94,0 % (UD-Q3_K_XL).

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#flint-40b-a25b).

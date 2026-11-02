# Kolibri 1

Retour à [README.fr.md](../../README.fr.md)

**Kolibri 1** d'Aleph Alpha, un modèle de 19,200 experts qui écrit l'allemand aussi bien que l'anglais : un fichier de
29 à 48 Go, qui reste surtout en RAM à côté d'une petite carte. Quatre couches sur cinq ne regardent que 513 tokens en
arrière.

## Quelle est sa vitesse ?

Mesuré comme [Flint 40B-A2.5B](flint.fr.md), avec un prompt de 31,676 tokens.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q4_K_M** | Non mesuré | Non mesuré |
| **Q2_K** | Non mesuré | Non mesuré |
| **Q3_K_M** | Non mesuré | Non mesuré |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q2_K** | 189-192 tokens/s | 4,980-5,070 tokens/s |

</td></tr>
</table>

La carte de la station de travail contient les 19,200 experts de Q2_K (25,6 Gio). Il a commencé à répondre au prompt de
31,676 tokens après 6,3 à 6,4 secondes, puis a écrit à 129-131 tokens/s. Q4_K_M et Q3_K_M n'y ont pas tourné, et
aucune taille n'a tourné sur le portable.

La pull request qui a ajouté le modèle l'a mesuré sur d'autres PC. Sur une RTX 3060 (12 Go) avec 24 Go de RAM sous
Windows, Q2_K a écrit un court chat à 21 tokens/s et lu un prompt de 32K à 698. Sur une Tesla V100 (32 Go), Q4_K_M a
écrit à 91 tokens/s et lu un prompt de 32K à 1,017 ; sur deux, il l'a lu à 2,179.

Face à llama.cpp avec le patch `kolibri1` de la conversion, en suivant sa réponse greedy à un prompt en anglais et à un
prompt en allemand, Q2_K choisit le même token le plus probable à 120 et 126 étapes sur 128. Là où ils diffèrent, le
token de llama.cpp est le deuxième ou le troisième choix de ce moteur à toutes les étapes sauf une, presque aussi
probable.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 4 Go sont pris avant le premier expert. Il a tourné sur une carte de 12 Go, une Tesla V100 de 32 Go et une RTX 5090 de 32 Go. |
| **RAM** | Environ **52, 33 ou 42 Go** à côté d'une carte de 8 Go pour Q4_K_M, Q2_K et Q3_K_M, d'après l'estimation de l'installeur. À côté d'une carte de 32 Go, Q2_K demande environ 10 Go et Q4_K_M environ 26 Go. |
| **Disque** | 29 à 48 Go pour le fichier du modèle. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur prépare tout le reste, comme pour [Qwen3.6-35B-A3B](qwen36.fr.md). Il n'a pas été essayé sur une carte AMD.
Il ne lit que du texte et n'a pas de modèle brouillon, donc il écrit un token à la fois. Sa licence est Apache 2.0.

## Quel modèle choisir ?

Trois tailles, issues de la conversion de [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF). À côté d'une
carte de 8 Go, l'installeur prend Q4_K_M à partir d'environ 48 Go de RAM et Q2_K en dessous ; à côté d'une carte plus
grande, les deux demandent moins. Q3_K_M se choisit par son nom : `--model Q3_K_M`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **Q4_K_M** | 4 bits, avec des projections down à 6 bits dans la moitié des couches | 48 Go | environ 52 Go |
| **Q2_K** | 2 bits : le moins de RAM | 29 Go | environ 33 Go |
| **Q3_K_M** | 3 bits | 38 Go | environ 42 Go |

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#kolibri-1).

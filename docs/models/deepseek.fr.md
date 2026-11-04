# DeepSeek V4 Flash

Retour à [README.fr.md](../../README.fr.md)

**DeepSeek V4 Flash**, un fichier de 91 Go, dont la plus grande part reste aussi en RAM. Sa mémoire d'une discussion est petite : un contexte de 128K ne prend que 0,7 Go de VRAM de plus que 32K.

## Quelle est sa vitesse ?

Mesuré pour PolyStrata sur la même station de travail que [GLM 5.3 Flash](glm.fr.md), avec une courte discussion et un prompt de 30K tokens.

<table>
<tr><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **UD-IQ2_XXS** | 56 tokens/s | 1,040 tokens/s |

</td></tr>
</table>

Sur le même PC, llama.cpp écrivait 25 à 28 tokens par seconde et lisait un prompt de 3,384 tokens à 96. Un prompt de
116,473 tokens, dans un contexte de 128K, a été lu et a reçu sa réponse en 174 secondes, un de 487,529 tokens, dans
un contexte de 512K, en 32 minutes, et les deux fois les trois faits cachés dedans ont été retrouvés. Ce qui a été mesuré et comment : [docs/DEEPSEEK.md](../DEEPSEEK.md#what-it-was-measured-on) (en anglais).

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente. Il n'a tourné qu'avec **32 Go de VRAM**. Environ 11,5 Go sont pris avant le premier expert : une carte de 12 Go ne peut donc pas le démarrer. |
| **RAM** | À côté d'une carte de 32 Go, **72 Go ou plus**, selon l'estimation de l'installeur. Une carte plus petite demande plus de RAM ([combien](#quel-modèle-choisir-)). Il a tourné avec 120 Go. |
| **Disque** | 91 Go pour les trois fichiers du modèle, sur un SSD. |
| **Système** | Linux, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste, comme pour GLM 5.3 Flash. Pas encore essayé : les cartes AMD, deux
cartes. Il ne lit que du texte.
Un utilisateur le fait tourner sous Windows avec une carte de 24 Go et 128 Go de RAM ([rapport](https://github.com/VecSzn/PolyStrata/issues/21)).

La liste complète : [docs/DEEPSEEK.md](../DEEPSEEK.md#what-you-need) (en anglais).

## Quel modèle choisir ?

Une seule taille pour l'instant.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 32 Go |
| --- | --- | ---: | ---: |
| **UD-IQ2_XXS** | le modèle entier, à environ 2 bits ([Unsloth](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)) | 91 Go | 72 Go |

Comme pour GLM 5.3 Flash, les experts que la carte ne garde pas restent en RAM : selon l'estimation de l'installeur,
environ 80 Go à côté d'une carte de 24 Go et 89 Go à côté d'une carte de 16 Go ; une carte de 12 Go ne peut pas le
démarrer. Seule la carte de 32 Go a été essayée.

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#deepseek-v4-flash).

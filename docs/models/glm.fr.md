# GLM 5.3 Flash

Retour à [README.fr.md](../../README.fr.md)

**GLM 5.3 Flash**, le plus grand : un fichier de 79 à 114 Go, dont la plus grande part reste en RAM.

## Quelle est sa vitesse ?

Mesuré pour PolyStrata sur une station de travail, avec une courte discussion et un prompt de 29K tokens.

<table>
<tr><th>NVIDIA : RTX 5090 (32 Go), Xeon Platinum 8470Q (25 vCPU), 120 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **3.0BIT** | 59-66 tokens/s | 820 tokens/s |
| **UD-IQ2_XXS** | 50-52 tokens/s | 1,060-1,080 tokens/s |
| **REAP50-Q3_K_M** | 59-66 tokens/s | 1,300-1,310 tokens/s |

</td></tr>
</table>

GLM 5.3 Flash est un modèle bien plus grand que les autres, et seul un cinquième environ de ses experts tient dans les 32 Go de cette carte :
il écrit donc plus lentement. Ce qui a été mesuré et comment : [docs/GLM.md](../GLM.md#what-it-was-measured-on) (en anglais).

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente. Il n'a tourné qu'avec **32 Go de VRAM**. Environ 12 Go sont pris avant le premier expert : une carte de 12 Go ne peut donc pas le démarrer. |
| **RAM** | À côté d'une carte de 32 Go, **96 Go ou plus** pour le modèle entier, 82 Go pour sa taille à 2 bits, 60 Go pour la taille qui garde la moitié des experts. Une carte plus petite demande plus de RAM ([combien](#quel-modèle-choisir-)). Il a tourné avec 120 Go. |
| **Disque** | 114 Go, 102 Go ou 79 Go pour le fichier du modèle, sur un SSD. |
| **Système** | Linux, et un pilote NVIDIA récent. |

L'installeur s'occupe de tout le reste. Il compile le moteur sur votre PC et propose d'installer ce qu'il faut pour cela
(g++ et le CUDA Toolkit). Pas encore : les cartes AMD, deux cartes sous Windows, trois cartes ou plus, les images.
Un utilisateur le fait tourner sous Windows avec une carte de 24 Go et 128 Go de RAM ([rapport](https://github.com/VecSzn/PolyStrata/issues/10)).

La liste complète : [docs/GLM.md](../GLM.md#what-you-need) (en anglais).

## Quel modèle choisir ?

Trois tailles pour l'instant. L'installeur prend la première pour laquelle votre RAM suffit.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 32 Go |
| --- | --- | ---: | ---: |
| **3.0BIT** | le modèle entier, à environ 3 bits ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 Go | 96 Go |
| **UD-IQ2_XXS** | le modèle entier, à environ 2 bits ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)) : il lit un prompt plus vite et écrit plus lentement | 102 Go | 82 Go |
| **REAP50-Q3_K_M** | la moitié des experts routés retirée ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) de patrickbdevaney) | 79 Go | 60 Go |

C'est votre carte graphique qui décide de la RAM nécessaire : les experts que la carte ne contient pas doivent rester en RAM.

| Votre carte graphique | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 Go** | **96 Go** | **82 Go** | **60 Go** | le PC sur lequel il a tourné (120 Go de RAM) |
| **24 Go** | environ 104 Go | environ 90 Go | environ 68 Go | pas encore essayé |
| **16 Go** | environ 113 Go | environ 99 Go | environ 77 Go | pas encore essayé |
| **12 Go ou moins** | | | | ne démarre pas |

Les chiffres de RAM sont l'estimation de l'installeur, d'après le seul PC où il a tourné. Avec moins de RAM il démarre quand
même, mais il lit des experts sur le SSD pendant qu'il répond et devient bien plus lent. L'installeur le dit et demande
d'abord.

Les autres quantifications de GLM 5.3 Flash ne sont pas encore prises en charge : [ce qui manque](../GLM.md#what-is-missing) (en anglais).

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#glm-53-flash).

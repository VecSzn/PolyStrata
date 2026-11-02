# Clover-35B-A3B-Preview

Retour à [README.fr.md](../../README.fr.md)

**Clover-35B-A3B-Preview** de DiffuSpace, un fichier de 13 à 21 Go. C'est Qwen3.6-35B-A3B entraîné plus loin pour
écrire aussi un bloc de tokens d'un coup. Il ne lit que du texte.

## Quelle est sa vitesse ?

Mesuré par PolyStrata sur le portable, avec une courte discussion et un prompt de 32,000 tokens. Sa carte a 8 Go, et le
processeur calcule les experts que la carte ne peut pas contenir.

<table>
<tr><th>NVIDIA : RTX 4070 Laptop (8 Go), Core i7-14700HX, 32 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q3_K_M** | 68-73 tokens/s | 1,560-1,700 tokens/s |
| **APEX-I-NANOPLUS** | 72-76 tokens/s | 1,530-1,690 tokens/s |
| **Q4_K_M** | 38-58 tokens/s | 1,220-1,360 tokens/s |

</td></tr>
</table>

Après le prompt de 32,000 tokens, le portable écrit à 60 à 65 tokens/s avec Q3_K_M. Q4_K_M y remplit les 32 Go de RAM,
ce qui le rend lent et irrégulier. La station de travail avec la RTX 5090 n'a pas été mesurée.

Qwen3.6-35B-A3B écrit plus vite sur le même portable, 94 à 100 tokens/s, parce que son fichier a un bloc de brouillon
qui devine les mots suivants pour le travail d'une couche. Les fichiers de Clover n'en ont pas. Clover peut deviner
lui-même un bloc de mots, mais cela refait tourner tout le modèle, et PolyStrata le laisse désactivé : sur une carte qui
contenait tous les experts, les réponses arrivaient plus tôt sans lui ([mesures](../BACKENDS.md#clover-35b-a3b-preview),
en anglais).

Sur une RTX PRO 6000 qui contenait tous les experts de Q3_K_M, PolyStrata a choisi le même token suivant que llama.cpp
(la prise en charge `clover_moe` de sa pull request 31306) à 127 étapes sur 128 après un prompt court, et à 125 sur
128 après 5,573 tokens.

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | Une carte **NVIDIA** GeForce RTX série 20 ou plus récente avec **8 Go de VRAM ou plus**. Environ 3,5 Go sont pris avant le premier expert. Il a tourné sur la carte de 8 Go d'un portable et sur une carte de 96 Go. Il n'a pas été essayé sur une carte AMD. |
| **RAM** | Environ **16, 20 ou 25 Go** à côté d'une carte de 8 Go pour APEX-I-NANOPLUS, Q3_K_M et Q4_K_M, d'après l'estimation de l'installeur. Une carte de 24 Go contient en entier les deux plus petites. Les trois ont tourné sur un portable de 32 Go. |
| **Disque** | 13, 17 ou 21 Go pour le fichier du modèle. |
| **Système** | Linux ou Windows 10 / 11, et un pilote NVIDIA récent. |

L'installeur s'occupe du reste, comme pour [Qwen3.6-35B-A3B](qwen36.fr.md). Les fichiers de maternion n'ont pas de
modèle de chat, donc l'installeur télécharge celui du modèle chez DiffuSpace. La licence de Clover est Apache 2.0.

## Quel modèle choisir ?

Trois tailles, toutes faites avec le convertisseur de la pull request 31306 de llama.cpp. L'installeur prend Q3_K_M,
ou APEX-I-NANOPLUS sur un PC qui a moins de RAM. Q4_K_M se choisit par son nom : `--model Q4_K_M`.

| Taille | Ce que c'est | Téléchargement | RAM à côté d'une carte de 8 Go |
| --- | --- | ---: | ---: |
| **Q3_K_M** | le modèle entier avec des experts à 3 bits ([maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)) | 17 Go | environ 20 Go |
| **APEX-I-NANOPLUS** | des experts à 2 et 3 bits, d'après une matrice d'importance ([IsValorum](https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF)) : celui qui demande le moins de RAM | 13 Go | environ 16 Go |
| **Q4_K_M** | des experts à 4 bits (maternion) | 21 Go | environ 25 Go |

Les deux tailles de maternion ont été quantifiées sans matrice d'importance. La distance de chaque taille au modèle
entraîné n'a pas été mesurée.

## Comment ça marche ?

Le schéma et le fonctionnement : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#clover-35b-a3b-preview).

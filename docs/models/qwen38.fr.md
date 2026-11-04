# Qwen3.8-Flash-Next

Retour à [README.fr.md](../../README.fr.md)

**Qwen3.8-Flash-Next** sur une carte graphique NVIDIA ou AMD de 12 Go ou plus, sous Windows ou Linux. Il lit aussi les images. Son moteur est celui de [Strata](https://github.com/Niko1221/Strata), inchangé.

## Quelle est sa vitesse ?

Mesuré par [Strata](https://github.com/Niko1221/Strata) sur deux PC de jeu ordinaires, avec un prompt de 32K tokens.

<table>
<tr><th>NVIDIA : RTX 5070 (12 Go), Ryzen 5 7600, 64 Go de RAM</th><th>AMD : RX 9070 XT (16 Go), Ryzen 9 3900X, 47 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA : Q2_0 avec le moteur 0.1.36, les autres lignes avec 0.1.26 (réponses de 4K, prompts de 32K). Les tableaux
complets sont dans [DETAILS.md](../DETAILS.md#speed-measured). Une carte avec plus de VRAM va plus vite : une
RTX 3090 (24 Go) devrait écrire environ 100-140 tokens par seconde. Longues discussions et autres cartes :
[vitesse de chaque modèle](../MODELS.md#how-fast-is-each-size), [résultats de la communauté](../COMMUNITY_BENCHMARKS.md).

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | **NVIDIA** GeForce RTX série 20, 30, 40 ou 50, ou **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 ou série RX 6800 / 6900. Il faut **12 Go de VRAM ou plus**. |
| **RAM** | 32 Go ou plus. Votre RAM décide [quel modèle](#quel-modèle-choisir-) tient. Avec 64 Go, toutes les tailles tournent. |
| **Disque** | Environ 80 Go libres. Prenez un SSD si possible : le premier démarrage est bien plus rapide. |
| **Système** | Windows 10 / 11 ou Linux, avec un pilote graphique NVIDIA ou AMD à jour. |

L'installateur s'occupe de tout le reste. Deux ou trois cartes peuvent se partager le modèle ([multi-GPU](../MULTI_GPU.md)).

Expérimental, écrit et testé par des membres de la communauté sur leurs propres machines :

- **Cartes graphiques plus anciennes** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT) : [Older GPUs](../OLDER_GPUS.md).
- **Intel Arc**, compilé depuis les sources sous Linux : [Intel Arc](../INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, compilé depuis les sources sous Linux : [Strix Halo](../STRIX_HALO.md).
- **Processeurs plus anciens sans AVX2** : ça marche, mais lentement. [Older CPUs](../INSTALL.md#older-cpus-experimental).

La liste complète : [docs/INSTALL.md](../INSTALL.md#what-you-need).

## Quel modèle choisir ?

L'installateur en recommande un selon votre RAM. Le même modèle existe en plusieurs tailles, plus ou moins
compressées. Les petites tailles sont plus rapides. Les grandes sont un peu plus intelligentes.

| Votre RAM | Prenez | Pourquoi |
| --- | --- | --- |
| **32 Go** | **Coder** | il tient dans 32 Go et il est fait pour le code (avec une carte de 24 Go, Q2_0 et IQ2_XS tournent aussi) |
| **48 Go** | **IQ2_XS** (ou Q2_0, le plus rapide) | les plus grandes tailles ne tiennent pas |
| **64 Go** | **IQ2_XS** (recommandé), ou IQ3_XXS / IQ3_S | toutes les tailles tiennent ; IQ3_S est le meilleur et le plus lent |
| **96 Go ou plus** | **IQ3_S**, ou l'UD-IQ4_XS d'Unsloth (~4 bits) | de la place pour les plus grandes tailles, avec tout le reste ouvert |

- **[Coder](../MODELS.md#coder) :** une version pour le code, avec la moitié des experts en moins. Elle atteint 91 %
  du score SWE-bench Verified du modèle complet (mesuré par ses auteurs) et tient dans 32 Go de RAM. Elle est moins
  bonne en dehors du code, y compris pour le chinois et les autres textes CJK (#438). Pour ces cas, prenez Q2_0,
  IQ2_XS ou IQ3_S, qui gardent tous les experts.
- **[Swift 1.5](../MODELS.md#swift-15) :** un fine-tune qui réfléchit bien moins longtemps avant de répondre. Vous
  avez la réponse plus tôt, avec à peu près la même qualité.
- **[Unsloth UD-IQ4_XS](../MODELS.md#unsloth-ud-iq4_xs)** : la version ~4 bits d'Unsloth, entre IQ3_S et
  UD-Q4_K_XL en qualité. Un téléchargement de 94 Go. Avec moins de ~80 Go de RAM, PolyStrata en lit une partie depuis
  le SSD pendant qu'il répond, il y est donc plus lent (un SSD NVMe aide).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL et Q8_0](../MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0) :** de 4 à 8 bits, plus proches du modèle
  complet (93-95 % le même token suivant que Q8_0). Rapides quand tous les experts tiennent dans la RAM (~88 Go pour
  UD-Q4_K_XL jusqu'à ~136 Go pour Q8_0) ; avec moins de RAM, PolyStrata en lit une partie depuis le SSD pendant qu'il
  répond, ce qui est bien plus lent (7-8.5 tokens/s pour UD-Q4_K_XL sur un PC avec 64 Go).
- **[OrcaRouter's Uncensored IQ3_XXS](../MODELS.md#orcarouter-uncensored-iq3_xxs) :** vous l'installez à la main.
  Il n'est pas dans le menu de l'installateur.

Tailles, téléchargements et ce qui tient où : [docs/MODELS.md](../MODELS.md). Pour ajouter un autre modèle plus
tard, lancez `SETUP.bat` (Linux : `./setup.sh --setup`).

## Comment ça marche ?

Le schéma et l'explication : [HOW_IT_WORKS_MODELS.fr.md](../HOW_IT_WORKS_MODELS.fr.md#qwen38-flash-next).

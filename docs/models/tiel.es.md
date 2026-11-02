# Tiel Coder 35B-A3B

Volver a [README.es.md](../../README.es.md)

**Tiel Coder 35B-A3B**, Ornith 1.5 cuantizado por peculiar-ragdoll para agentes de programación: un archivo de 14 a 23 GB. También entiende imágenes.

## ¿Qué velocidad tiene?

Medido igual que [Qwen3.6-35B-A3B](qwen36.es.md), en los mismos dos PC.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 83 tokens/s | 1,700 tokens/s |
| **UD-IQ3_XXS** | 96-97 tokens/s | 1,610 tokens/s |
| **UD-Q4_K_XL** | 74-79 tokens/s | 1,660-1,670 tokens/s |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 337-345 tokens/s | 8,080-8,110 tokens/s |
| **UD-IQ3_XXS** | 352 tokens/s | 7,520 tokens/s |
| **UD-Q4_K_XL** | 316-322 tokens/s | 7,750-7,970 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens se lee a 1,670 tokens/s en el portátil y a 6,300 en la estación de trabajo.

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 5 GB se ocupan antes del primer experto. Ha funcionado en una tarjeta de 32 GB, en esa tarjeta limitada a 8 GB y en la tarjeta de 8 GB de un portátil. En una tarjeta AMD también lo ejecuta, en beta, en Linux y en Windows: el instalador compila el motor en tu PC; hay usuarios que lo han ejecutado en una RX 6700 XT y una RX 7800 XT en Linux, y en una RX 9070 XT en Windows. |
| **RAM** | Unos **18, 21 o 27 GB** junto a una tarjeta de 8 GB para los tres tamaños, según la estimación del instalador. Una tarjeta de 24 GB guarda enteros los tres tamaños. Los tres tamaños han funcionado en un portátil con 32 GB. |
| **Disco** | 14, 17 o 23 GB para el archivo del modelo, 1 GB más para leer imágenes. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás, como con GLM 5.3 Flash. En Windows descarga el motor ya compilado de la
última versión publicada. En un procesador sin AVX2, o en una copia cuyo motor cambió después de esa versión, lo
compila en su lugar con el compilador de 2022 de Visual Studio y el CUDA Toolkit, y ofrece instalarlos. Todavía no:
dos tarjetas en Windows, tres tarjetas o más. En una tarjeta AMD bajo Windows el instalador compila `strata-poly` para
la tarjeta, como en Linux (las wheels TheRock de AMD en `.rocm-win`), y avisa de que ejecutarlo allí es nuevo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## ¿Qué modelo elijo?

Tres tamaños por ahora. El instalador toma UD-Q3_K_XL, o UD-IQ3_XXS en un PC con menos RAM. UD-Q4_K_XL se elige por su nombre: `--model UD-Q4_K_XL`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q3_K_XL** | el modelo entero con expertos de 3 y 4 bits ([peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)) | 17 GB | unos 21 GB |
| **UD-IQ3_XXS** | el mismo con expertos de 2 y 3 bits: el que menos RAM pide | 14 GB | unos 18 GB |
| **UD-Q4_K_XL** | el mismo con expertos de 4 a 6 bits | 23 GB | unos 27 GB |

Una tarjeta más grande guarda más parte del modelo y necesita menos RAM: con UD-Q3_K_XL, unos 17 GB junto a una tarjeta de 12 GB, 13 GB junto a una de 16 GB y 10 GB cuando la tarjeta lo guarda todo. Son estimaciones del instalador. No se
han comprobado en un PC así.

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#tiel-coder-35b-a3b).

# Ornith 1.5

Volver a [README.es.md](../../README.es.md)

**Ornith 1.5**, con la arquitectura de Qwen3.6-35B-A3B y otros pesos: un archivo de 14 a 24 GB. También entiende imágenes.

## ¿Qué velocidad tiene?

Medido igual que [Qwen3.6-35B-A3B](qwen36.es.md), en los mismos dos PC.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 76-82 tokens/s | 1,620 tokens/s |
| **APEX-I-MINI** | 80-86 tokens/s | 1,550-1,560 tokens/s |
| **APEX-I-QUALITY** | 59-63 tokens/s | 1,570-1,580 tokens/s |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 316-319 tokens/s | 8,040-8,110 tokens/s |
| **APEX-I-MINI** | 289-300 tokens/s | 7,710-7,740 tokens/s |
| **APEX-I-QUALITY** | 297-302 tokens/s | 7,860-7,880 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens se lee a 1,590 tokens/s en el portátil y a 6,400 en la estación de trabajo.

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 4 GB se ocupan antes del primer experto. Ha funcionado en una tarjeta de 32 GB, en esa tarjeta limitada a 8 GB y en la tarjeta de 8 GB de un portátil. En una tarjeta AMD también lo ejecuta, en beta, en Linux y en Windows: el instalador compila el motor en tu PC; probado hasta ahora con Qwen3.6 y Ornith en una RX 7600. |
| **RAM** | Unos **18, 22 o 28 GB** junto a una tarjeta de 8 GB para los tres tamaños, según la estimación del instalador. Una tarjeta de 24 GB guarda enteros los dos tamaños más pequeños. Los tres tamaños han funcionado en un portátil con 32 GB. |
| **Disco** | 14, 17 o 24 GB para el archivo del modelo, 1 GB más para leer imágenes. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás, como con GLM 5.3 Flash. En Windows descarga el motor ya compilado de la
última versión publicada. En un procesador sin AVX2, o en una copia cuyo motor cambió después de esa versión, lo
compila en su lugar con el compilador de 2022 de Visual Studio y el CUDA Toolkit, y ofrece instalarlos. Todavía no:
dos tarjetas en Windows, tres tarjetas o más. En una tarjeta AMD bajo Windows el instalador compila `strata-poly` para
la tarjeta, como en Linux (las wheels TheRock de AMD en `.rocm-win`), y avisa de que ejecutarlo allí es nuevo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## ¿Qué modelo elijo?

Tres tamaños por ahora. El instalador toma APEX-I-COMPACT, o APEX-I-MINI en un PC con menos RAM. APEX-I-QUALITY se elige por su nombre: `--model APEX-I-QUALITY`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | el modelo entero con expertos de 3 y 4 bits ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) de mudler) | 17 GB | unos 22 GB |
| **APEX-I-MINI** | el mismo con expertos de 2 y 3 bits: el que menos RAM pide | 14 GB | unos 18 GB |
| **APEX-I-QUALITY** | el mismo con expertos de 4 a 6 bits | 24 GB | unos 28 GB |

Una tarjeta más grande guarda más parte del modelo y necesita menos RAM: con APEX-I-COMPACT, unos 17 GB junto a una tarjeta de 12 GB, 13 GB junto a una de 16 GB y 10 GB cuando la tarjeta lo guarda todo. Son estimaciones del instalador. No se
han comprobado en un PC así.

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#ornith-15).

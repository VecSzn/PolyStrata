# Flint 40B-A2.5B

Volver a [README.es.md](../../README.es.md)

**Flint 40B-A2.5B** de Recursive, hecho para agentes de código: un archivo de 18 a 42 GB. La mayoría de sus capas solo mira 513 tokens atrás, así que una conversación larga ocupa poca VRAM: un contexto de 128K solo ocupa 2,6 GB más que 32K.

## ¿Qué velocidad tiene?

Medido como [LFM2.5-24B-A2B](lfm25.es.md), en los mismos dos PC, con un prompt de unos 29,000 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 90 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 20-29 tokens/s | 1,310-1,420 tokens/s |
| **UD-Q3_K_XL** | 61-63 tokens/s | 2,000-2,010 tokens/s |
| **UD-Q6_K_XL** | No cabe | No cabe |
| **Q8_0** | No cabe | No cabe |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 251-252 tokens/s | 12,900 tokens/s |
| **UD-Q3_K_XL** | 260-264 tokens/s | 13,080-13,200 tokens/s |
| **Q8_0** | 158-160 tokens/s | 6,740-6,970 tokens/s |

</td></tr>
</table>

En el portátil solo UD-Q3_K_XL cabe en los 32 GB de RAM junto a los programas del escritorio. Los tamaños mayores
vuelven a leer sus expertos del SSD una y otra vez, y por eso van lentos allí; el instalador toma allí UD-Q3_K_XL. La
tarjeta de la estación de trabajo guarda enteros UD-Q4_K_XL y UD-Q3_K_XL; UD-Q6_K_XL no se ejecutó allí.

En el portátil llama.cpp con `--fit on`, el más rápido de los ajustes probados allí, leyó un prompt de 29,442 tokens
a 390 tokens/s y escribió después a 35; PolyStrata lo leyó a 1,950-2,010 y escribió a 49-52
([las mediciones](../../bench/results/2026-10-24-laptop-flint-llamacpp-a37cc46/README.md)).

En la estación de trabajo llama-bench de llama.cpp leyó 29,816 tokens a 7,790 tokens/s con su lote por defecto de 512
y a 12,470 con 2,048, PolyStrata a 12,910-12,940; token a token llama.cpp escribió a 222 tokens/s y a 188 después de ese
prompt, PolyStrata a 249-256 y 212-213. Un prompt de 124,122 tokens en un contexto de 128K se leyó y respondió en 20
segundos, y se encontraron los tres datos escondidos en él.

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 4 GB se ocupan antes del primer experto. Ha funcionado en la tarjeta de 8 GB de un portátil y en una tarjeta de 32 GB. |
| **RAM** | Unos **22, 29, 40 o 46 GB** junto a una tarjeta de 8 GB para UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL y Q8_0, según la estimación del instalador. En un portátil con 32 GB solo UD-Q3_K_XL se quedó en la RAM, así que el instalador toma UD-Q4_K_XL a partir de unos 40 GB. Una tarjeta de 32 GB guarda entero UD-Q4_K_XL. |
| **Disco** | De 18 a 42 GB para el archivo del modelo. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás, como con [Qwen3.6-35B-A3B](qwen36.es.md). Aún no se ha probado en una tarjeta
AMD. Solo lee texto. Su licencia es Apache 2.0.

La lista completa: [docs/FLINT.md](../FLINT.md#what-you-need) (en inglés).

## ¿Qué modelo elijo?

Cuatro tamaños. Junto a una tarjeta de 8 GB el instalador toma UD-Q4_K_XL a partir de unos 40 GB de RAM y UD-Q3_K_XL
por debajo; junto a una tarjeta mayor UD-Q4_K_XL necesita menos. Los demás se eligen por su nombre: `--model Q8_0`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | el modelo entero con expertos de 4 y 5 bits ([Unsloth](https://huggingface.co/unsloth/flint-GGUF)) | 25 GB | unos 29 GB |
| **UD-Q3_K_XL** | lo mismo con expertos de 3 bits: la menor RAM | 18 GB | unos 22 GB |
| **UD-Q6_K_XL** | expertos de 6 bits | 36 GB | unos 40 GB |
| **Q8_0** | 8 bits en todo, el más cercano al modelo tal como se entrenó | 42 GB | unos 46 GB |

En el portátil de 32 GB el archivo de UD-Q4_K_XL no se quedó en la RAM: escribió un chat corto a 20-29 tokens/s,
UD-Q3_K_XL a 61-63. Frente a Q8_0, con cinco prompts de 256 tokens cada uno, los otros tamaños eligen el token más
probable de Q8_0 en el 97,8 % de los pasos (UD-Q6_K_XL), el 95,9 % (UD-Q4_K_XL) y el 94,0 % (UD-Q3_K_XL).

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#flint-40b-a25b).

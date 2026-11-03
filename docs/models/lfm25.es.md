# LFM2.5-24B-A2B

Volver a [README.es.md](../../README.es.md)

**LFM2.5-24B-A2B** de Liquid AI, un archivo de 11 a 25 GB. La mayoría de sus capas no tiene atención, así que una conversación larga ocupa poca VRAM: un contexto de 128K solo ocupa 2 GB más que 32K. Su licencia solo permite el uso comercial a empresas que facturan menos de 10 millones de dólares al año.

## ¿Qué velocidad tiene?

Medido como [Qwen3.6-35B-A3B](qwen36.es.md), en los mismos dos PC, con un prompt de unos 30,000 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 70-73 tokens/s | 3,590-3,620 tokens/s |
| **UD-Q3_K_XL** | 88 tokens/s | 3,760-3,810 tokens/s |
| **QAD-Q4_0** | 77-78 tokens/s | 3,720-3,730 tokens/s |
| **UD-Q6_K_XL** | 42-43 tokens/s | 3,030-3,120 tokens/s |
| **Q8_0** | 23-34 tokens/s | 2,950-3,040 tokens/s |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 480-485 tokens/s | 23,390-23,460 tokens/s |
| **UD-Q3_K_XL** | 490-497 tokens/s | 22,920-23,400 tokens/s |
| **QAD-Q4_0** | 498 tokens/s | 23,880-23,900 tokens/s |
| **UD-Q6_K_XL** | 423-426 tokens/s | 21,250-21,270 tokens/s |
| **Q8_0** | 376 tokens/s | 23,270-23,350 tokens/s |

</td></tr>
</table>

En el portátil Q8_0 llena los 32 GB de RAM, y por eso va lento allí; la estación de trabajo guarda entero cada tamaño.

En el portátil llama.cpp b11540 con `--n-cpu-moe 25`, el mejor de los ajustes probados allí, leyó un prompt de 32,000
tokens a 530 tokens/s y escribió después a 53; PolyStrata lo leyó a 3,540 y escribió a 71-73. En la estación de
trabajo llama-bench leyó 29,816 tokens a 13,300 tokens/s con el lote por defecto de llama.cpp, de 512, y a 23,600 con
2,048, tan rápido como PolyStrata, y escribió token a token a 455 tokens/s, y a 369 después de ese prompt; PolyStrata
escribe a 452 y 387. Un prompt de unos 124,000 tokens en un contexto de 128K se leyó y respondió en 8,7 segundos en la
estación de trabajo; el portátil lo leyó en 45 segundos.

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 4 GB se ocupan antes del primer experto. Ha funcionado en la tarjeta de 8 GB de un portátil y en una tarjeta de 32 GB. |
| **RAM** | Unos **16, 18, 20, 27 o 31 GB** junto a una tarjeta de 8 GB para UD-Q3_K_XL, QAD-Q4_0, UD-Q4_K_XL, UD-Q6_K_XL y Q8_0, según la estimación del instalador. Una tarjeta de 24 GB guarda enteros los tres tamaños más pequeños. Los cinco han funcionado en un portátil con 32 GB. |
| **Disco** | De 11 a 25 GB para el archivo del modelo. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás, como con [Qwen3.6-35B-A3B](qwen36.es.md). Un usuario lo ha ejecutado en una AMD
RX 6700 XT (12 GB) en Linux. Solo lee texto. La licencia de Liquid AI solo permite el uso comercial a empresas que facturan menos de
10 millones de dólares al año ([LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)).

La lista completa: [docs/LFM.md](../LFM.md#what-you-need) (en inglés).

## ¿Qué modelo elijo?

Cinco tamaños. El instalador toma UD-Q4_K_XL, o UD-Q3_K_XL en un PC con menos RAM. Los demás se eligen por su nombre: `--model QAD-Q4_0`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | el modelo entero con expertos de 4 bits ([Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)) | 15 GB | unos 20 GB |
| **UD-Q3_K_XL** | el mismo con expertos de 3 bits: el que menos RAM pide | 11 GB | unos 16 GB |
| **QAD-Q4_0** | 4 bits, de la propia [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF), entrenado para perder menos con esa cuantización | 14 GB | unos 18 GB |
| **UD-Q6_K_XL** | expertos de 6 bits | 22 GB | unos 27 GB |
| **Q8_0** | 8 bits en todo, el más cercano al modelo tal como se entrenó | 25 GB | unos 31 GB |

Frente a Q8_0, con cinco prompts de 256 tokens cada uno, los otros tamaños eligen el token más probable de Q8_0 en el
96,8 % de los pasos (UD-Q6_K_XL), el 95,5 % (UD-Q4_K_XL), el 92,7 % (QAD-Q4_0) y el 91,6 % (UD-Q3_K_XL).

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#lfm25-24b-a2b).

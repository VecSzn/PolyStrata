# GLM 5.3 Flash

Volver a [README.es.md](../../README.es.md)

**GLM 5.3 Flash**, el más grande: un archivo de 79 a 114 GB, que en su mayor parte se queda en la RAM.

## ¿Qué velocidad tiene?

Medido para PolyStrata en una estación de trabajo, con un chat corto y un prompt de 29K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **3.0BIT** | 59-66 tokens/s | 820 tokens/s |
| **UD-IQ2_XXS** | 50-52 tokens/s | 1,060-1,080 tokens/s |
| **REAP50-Q3_K_M** | 59-66 tokens/s | 1,300-1,310 tokens/s |

</td></tr>
</table>

GLM 5.3 Flash es un modelo mucho más grande que los demás, y solo una quinta parte de sus expertos cabe en los 32 GB de esta tarjeta, por eso
escribe más despacio. Qué se midió y cómo: [docs/GLM.md](../GLM.md#what-it-was-measured-on) (en inglés).

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva. Solo ha funcionado con **32 GB de VRAM**. Unos 12 GB se ocupan antes del primer experto, así que una tarjeta de 12 GB no puede arrancarlo. |
| **RAM** | Junto a una tarjeta de 32 GB, **96 GB o más** para el modelo entero, 82 GB para su tamaño de 2 bits, 60 GB para el tamaño con la mitad de los expertos. Una tarjeta más pequeña necesita más RAM ([cuánta](#qué-modelo-elijo)). Ha funcionado con 120 GB. |
| **Disco** | 114 GB, 102 GB o 79 GB para el archivo del modelo, en un SSD. |
| **Sistema** | Linux y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás. Compila el motor en tu PC y ofrece instalar lo que haga falta para eso
(g++ y el CUDA Toolkit). Todavía no: tarjetas AMD, dos tarjetas en Windows, tres tarjetas o más, imágenes.
Un usuario lo ejecuta en Windows con una tarjeta de 24 GB y 128 GB de RAM ([informe](https://github.com/VecSzn/PolyStrata/issues/10)).

La lista completa: [docs/GLM.md](../GLM.md#what-you-need) (en inglés).

## ¿Qué modelo elijo?

Tres tamaños por ahora. El instalador toma el primero para el que alcanza tu RAM.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 32 GB |
| --- | --- | ---: | ---: |
| **3.0BIT** | el modelo entero, a unos 3 bits ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 GB | 96 GB |
| **UD-IQ2_XXS** | el modelo entero, a unos 2 bits ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): lee un prompt más rápido y escribe más despacio | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | la mitad de los expertos enrutados eliminada ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) de patrickbdevaney) | 79 GB | 60 GB |

La RAM que necesita depende de tu tarjeta gráfica: los expertos que no caben en la tarjeta tienen que quedarse en la RAM.

| Tu tarjeta gráfica | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | el PC donde ha funcionado (120 GB de RAM) |
| **24 GB** | unos 104 GB | unos 90 GB | unos 68 GB | todavía sin probar |
| **16 GB** | unos 113 GB | unos 99 GB | unos 77 GB | todavía sin probar |
| **12 GB o menos** | | | | no arranca |

Las cifras de RAM son la estimación del instalador, a partir del único PC donde ha funcionado. Con menos RAM arranca
igualmente, pero lee expertos del SSD mientras responde y va mucho más lento. El instalador lo dice y pregunta antes.

Las demás cuantizaciones de GLM 5.3 Flash todavía no se admiten: [qué falta](../GLM.md#what-is-missing) (en inglés).

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#glm-53-flash).

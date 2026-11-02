# Kolibri 1

Volver a [README.es.md](../../README.es.md)

**Kolibri 1** de Aleph Alpha, un modelo de 19,200 expertos que escribe alemán tan bien como inglés: un archivo de 29 a
48 GB, que junto a una tarjeta pequeña queda casi todo en la RAM. Cuatro de cada cinco capas miran solo 513 tokens atrás.

## ¿Qué velocidad tiene?

Medido como [Flint 40B-A2.5B](flint.es.md), con un prompt de 31,676 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **Q4_K_M** | No medido | No medido |
| **Q2_K** | No medido | No medido |
| **Q3_K_M** | No medido | No medido |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **Q2_K** | 189-192 tokens/s | 4,980-5,070 tokens/s |

</td></tr>
</table>

La tarjeta de la estación de trabajo guarda los 19,200 expertos de Q2_K (25,6 GiB). Empezó a responder el prompt de
31,676 tokens a los 6,3 a 6,4 segundos y después escribió a 129-131 tokens/s. Q4_K_M y Q3_K_M no se ejecutaron allí, y
en el portátil no se ejecutó ningún tamaño.

El pull request que añadió el modelo lo midió en otros PC. En una RTX 3060 (12 GB) con 24 GB de RAM en Windows, Q2_K
escribió un chat corto a 21 tokens/s y leyó un prompt de 32K a 698. En una Tesla V100 (32 GB), Q4_K_M escribió a 91
tokens/s y leyó un prompt de 32K a 1,017; con dos leyó a 2,179.

Frente a llama.cpp con el parche `kolibri1` de la conversión, siguiendo su respuesta greedy a un prompt en inglés y otro
en alemán, Q2_K elige el mismo token más probable en 120 y 126 de 128 pasos. Donde difieren, el token de llama.cpp es
la segunda o tercera opción de este motor en todos los pasos menos uno, casi igual de probable.

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 4 GB se ocupan antes del primer experto. Ha funcionado en una tarjeta de 12 GB, una Tesla V100 de 32 GB y una RTX 5090 de 32 GB. |
| **RAM** | Unos **52, 33 o 42 GB** junto a una tarjeta de 8 GB para Q4_K_M, Q2_K y Q3_K_M, según la estimación del instalador. Junto a una tarjeta de 32 GB, Q2_K necesita unos 10 GB y Q4_K_M unos 26 GB. |
| **Disco** | De 29 a 48 GB para el archivo del modelo. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador prepara todo lo demás, como para [Qwen3.6-35B-A3B](qwen36.es.md). No se ha probado en una tarjeta AMD. Solo
lee texto y no tiene modelo borrador, así que escribe un token cada vez. Su licencia es Apache 2.0.

## ¿Qué modelo elijo?

Tres tamaños, de la conversión de [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF). Junto a una tarjeta de
8 GB el instalador toma Q4_K_M a partir de unos 48 GB de RAM y Q2_K por debajo; junto a una tarjeta mayor ambos
necesitan menos. Q3_K_M se elige por su nombre: `--model Q3_K_M`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **Q4_K_M** | 4 bits, con proyecciones down de 6 bits en la mitad de las capas | 48 GB | unos 52 GB |
| **Q2_K** | 2 bits: la menor RAM | 29 GB | unos 33 GB |
| **Q3_K_M** | 3 bits | 38 GB | unos 42 GB |

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#kolibri-1).

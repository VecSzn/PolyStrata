# Clover-35B-A3B-Preview

Volver a [README.es.md](../../README.es.md)

**Clover-35B-A3B-Preview** de DiffuSpace, un archivo de 13 a 21 GB. Es Qwen3.6-35B-A3B entrenado más para escribir
también un bloque de tokens de una vez. Solo lee texto.

## ¿Qué velocidad tiene?

Medido por PolyStrata en el portátil, con un chat corto y un prompt de 32,000 tokens. Su tarjeta tiene 8 GB, y el
procesador calcula los expertos que no caben en la tarjeta.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **Q3_K_M** | 68-73 tokens/s | 1,560-1,700 tokens/s |
| **APEX-I-NANOPLUS** | 72-76 tokens/s | 1,530-1,690 tokens/s |
| **Q4_K_M** | 38-58 tokens/s | 1,220-1,360 tokens/s |

</td></tr>
</table>

Después del prompt de 32,000 tokens, el portátil escribe a 60-65 tokens/s con Q3_K_M. Q4_K_M llena allí los 32 GB de
RAM, y eso lo hace lento e irregular. La estación de trabajo con la RTX 5090 no se midió.

Qwen3.6-35B-A3B escribe más rápido en el mismo portátil, 94 a 100 tokens/s, porque su archivo tiene un bloque de
borrador que adivina las palabras siguientes con el trabajo de una capa. Los archivos de Clover no lo tienen. Clover
puede adivinar él mismo un bloque de palabras, pero eso vuelve a ejecutar el modelo entero, y PolyStrata lo deja
apagado: en una tarjeta que guardaba todos los expertos, las respuestas llegaron antes sin él
([medido](../BACKENDS.md#clover-35b-a3b-preview), en inglés).

En una RTX PRO 6000 que guardaba todos los expertos de Q3_K_M, PolyStrata eligió el mismo token siguiente que llama.cpp
(el soporte `clover_moe` de su pull request 31306) en 127 de 128 pasos después de un prompt corto, y en 125 de 128
después de 5,573 tokens.

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 3,5 GB se ocupan antes del primer experto. Ha funcionado en la tarjeta de 8 GB de un portátil y en una tarjeta de 96 GB. No se ha probado en una tarjeta AMD. |
| **RAM** | Unos **16, 20 o 25 GB** junto a una tarjeta de 8 GB para APEX-I-NANOPLUS, Q3_K_M y Q4_K_M, según la estimación del instalador. Una tarjeta de 24 GB guarda enteros los dos tamaños más pequeños. Los tres han funcionado en un portátil con 32 GB. |
| **Disco** | 13, 17 o 21 GB para el archivo del modelo. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador prepara todo lo demás, como para [Qwen3.6-35B-A3B](qwen36.es.md). Los archivos de maternion no traen
plantilla de chat, así que el instalador descarga la del propio modelo desde DiffuSpace. La licencia de Clover es
Apache 2.0.

## ¿Qué modelo elijo?

Tres tamaños, todos hechos con el conversor de la pull request 31306 de llama.cpp. El instalador toma Q3_K_M, o
APEX-I-NANOPLUS en un PC con menos RAM. Q4_K_M se elige por su nombre: `--model Q4_K_M`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **Q3_K_M** | el modelo entero con expertos de 3 bits ([maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)) | 17 GB | unos 20 GB |
| **APEX-I-NANOPLUS** | expertos de 2 y 3 bits, según una matriz de importancia ([IsValorum](https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF)): el que menos RAM pide | 13 GB | unos 16 GB |
| **Q4_K_M** | expertos de 4 bits (maternion) | 21 GB | unos 25 GB |

Los dos tamaños de maternion se cuantizaron sin matriz de importancia. No se midió cuánto se acerca cada tamaño al
modelo entrenado.

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#clover-35b-a3b-preview).

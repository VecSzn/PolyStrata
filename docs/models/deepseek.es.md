# DeepSeek V4 Flash

Volver a [README.es.md](../../README.es.md)

**DeepSeek V4 Flash**, un archivo de 91 GB, que en su mayor parte también se queda en la RAM. Su memoria de una conversación es pequeña: un contexto de 128K ocupa solo 0,7 GB más de VRAM que uno de 32K.

## ¿Qué velocidad tiene?

Medido para PolyStrata en la misma estación de trabajo que [GLM 5.3 Flash](glm.es.md), con un chat corto y un prompt de 30K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-IQ2_XXS** | 56 tokens/s | 1,040 tokens/s |

</td></tr>
</table>

En el mismo PC, llama.cpp escribía de 25 a 28 tokens por segundo y leía un prompt de 3,384 tokens a 96. Un prompt de
116,473 tokens, en un contexto de 128K, se leyó y se respondió en 174 segundos, y uno de 487,529 tokens, en un
contexto de 512K, en 32 minutos; las dos veces se encontraron los tres datos escondidos en él. Qué se midió y cómo: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-it-was-measured-on) (en inglés).

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva. Solo ha funcionado con **32 GB de VRAM**. Unos 11,5 GB se ocupan antes del primer experto, así que una tarjeta de 12 GB no puede arrancarlo. |
| **RAM** | Junto a una tarjeta de 32 GB, **72 GB o más**, según la estimación del instalador. Una tarjeta más pequeña necesita más RAM ([cuánta](#qué-modelo-elijo)). Ha funcionado con 120 GB. |
| **Disco** | 91 GB para los tres archivos del modelo, en un SSD. |
| **Sistema** | Linux y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás, como con GLM 5.3 Flash. Aún sin probar: tarjetas AMD, dos
tarjetas. Solo lee texto.
Un usuario lo ejecuta en Windows con una tarjeta de 24 GB y 128 GB de RAM ([informe](https://github.com/VecSzn/PolyStrata/issues/21)).

La lista completa: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-you-need) (en inglés).

## ¿Qué modelo elijo?

Un solo tamaño por ahora.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 32 GB |
| --- | --- | ---: | ---: |
| **UD-IQ2_XXS** | el modelo entero, a unos 2 bits ([Unsloth](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)) | 91 GB | 72 GB |

Como con GLM 5.3 Flash, los expertos que la tarjeta no guarda se quedan en la RAM: según la estimación del instalador,
unos 80 GB junto a una tarjeta de 24 GB y 89 GB junto a una de 16 GB; una tarjeta de 12 GB no puede arrancarlo. Solo se
ha probado la tarjeta de 32 GB.

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#deepseek-v4-flash).

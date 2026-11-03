# MiniCPM-V 4.7 35B-A3B

Volver a [README.es.md](../../README.es.md)

**MiniCPM-V 4.7 35B-A3B** de OpenBMB, con la arquitectura de Qwen3.6-35B-A3B y pesos propios, hecho para leer imágenes: un archivo de 16 a 29 GB.

## ¿Qué velocidad tiene?

Medido igual que [Qwen3.6-35B-A3B](qwen36.es.md), en los mismos dos PC. Su archivo no tiene bloque borrador, así que escribe un token cada vez.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 64 tokens/s | 1,150 tokens/s |
| **mix_q3gud_im** | 59 tokens/s | 1,130-1,150 tokens/s |
| **Q6_K** | 40 tokens/s | 960-980 tokens/s |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 308-311 tokens/s | 7,910-7,950 tokens/s |
| **mix_q3gud_im** | 291-293 tokens/s | 7,730-7,740 tokens/s |
| **Q6_K** | 277-278 tokens/s | 7,290-7,330 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens se lee a 1,260 tokens/s en el portátil y a 8,790 en la estación de trabajo. En el portátil
Q6_K llena los 32 GB de RAM, y por eso allí es lento.

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 3.5 GB se ocupan antes del primer experto. Ha funcionado en una tarjeta de 32 GB y en la tarjeta de 8 GB de un portátil. En una tarjeta AMD también lo ejecuta, en beta, en Linux y en Windows: el instalador compila el motor en tu PC; ha funcionado en una RX 7600. |
| **RAM** | Unos **19, 25 o 31 GB** junto a una tarjeta de 8 GB para los tres tamaños, según la estimación del instalador. Una tarjeta de 32 GB guarda enteros los tres tamaños. Los tres tamaños han funcionado en un portátil con 32 GB. |
| **Disco** | 16, 21 o 29 GB para el archivo del modelo, 1 GB más para leer imágenes. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás, como con GLM 5.3 Flash. En Windows descarga el motor ya compilado de la
última versión publicada. En un procesador sin AVX2, o en una copia cuyo motor cambió después de esa versión, lo
compila en su lugar con el compilador de 2022 de Visual Studio y el CUDA Toolkit, y ofrece instalarlos. Todavía no:
dos tarjetas en Windows, tres tarjetas o más. En una tarjeta AMD bajo Windows el instalador compila `strata-poly` para
la tarjeta, como en Linux (las wheels TheRock de AMD en `.rocm-win`), y avisa de que ejecutarlo allí es nuevo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## ¿Qué modelo elijo?

Tres tamaños por ahora, de OpenBMB. El instalador toma Q4_K_M, o mix_q3gud_im en un PC con menos RAM. Q6_K se elige por su nombre: `--model Q6_K`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **Q4_K_M** | el modelo entero con expertos de 4 bits ([OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)) | 21 GB | unos 25 GB |
| **mix_q3gud_im** | el mismo con expertos de 3 y 4 bits: el que menos RAM pide | 16 GB | unos 19 GB |
| **Q6_K** | el mismo con expertos de 6 bits | 29 GB | unos 31 GB |

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#minicpm-v-47-35b-a3b).

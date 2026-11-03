# Qwen3.6-35B-A3B

Volver a [README.es.md](../../README.es.md)

**Qwen3.6-35B-A3B**, un archivo de 13 a 23 GB. También entiende imágenes.

## ¿Qué velocidad tiene?

Medido por PolyStrata en dos PC, con un chat corto y un prompt de 30,000 a 32,000 tokens: a la izquierda un portátil
con una tarjeta de 8 GB, donde el procesador calcula los expertos que no caben en la tarjeta, y a la derecha una
estación de trabajo cuya tarjeta tiene el modelo entero.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 94-100 tokens/s | 1,690 tokens/s |
| **UD-Q2_K_XL** | 123-124 tokens/s | 1,590 tokens/s |
| **UD-Q4_K_M** | 82-86 tokens/s | 1,560-1,570 tokens/s |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 394-395 tokens/s | 7,800-7,810 tokens/s |
| **UD-Q2_K_XL** | 450 tokens/s | 7,720-7,740 tokens/s |
| **UD-Q4_K_M** | 395-396 tokens/s | 8,070-8,080 tokens/s |

</td></tr>
</table>

Un prompt de 10,000 tokens se lee a 1,660 tokens/s en el portátil y a 6,300 en la estación de trabajo.
Las cifras de la tarjeta de la
estación limitada a 8 GB, un sustituto de antes de medir en el portátil, están en
[docs/BACKENDS.md](../BACKENDS.md#with-8-gb-of-vram) (en inglés).

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | Una tarjeta **NVIDIA** GeForce RTX serie 20 o más nueva con **8 GB de VRAM o más**. Unos 5 GB se ocupan antes del primer experto. Ha funcionado en una tarjeta de 32 GB, en esa tarjeta limitada a 8 GB y en la tarjeta de 8 GB de un portátil. En una tarjeta AMD también lo ejecuta, en beta, en Linux y en Windows: el instalador compila el motor en tu PC; probado hasta ahora con Qwen3.6 y Ornith en una RX 7600. |
| **RAM** | Unos **17, 21 o 27 GB** junto a una tarjeta de 8 GB para los tres tamaños, según la estimación del instalador. Una tarjeta de 24 GB guarda enteros los dos tamaños más pequeños. Los tres tamaños han funcionado en un portátil con 32 GB. |
| **Disco** | 13, 17 o 23 GB para el archivo del modelo, 1 GB más para leer imágenes. |
| **Sistema** | Linux o Windows 10 / 11, y un controlador de NVIDIA actual. |

El instalador se encarga de todo lo demás, como con GLM 5.3 Flash. En Windows descarga el motor ya compilado de la
última versión publicada. En un procesador sin AVX2, o en una copia cuyo motor cambió después de esa versión, lo
compila en su lugar con el compilador de 2022 de Visual Studio y el CUDA Toolkit, y ofrece instalarlos. Todavía no:
dos tarjetas en Windows, tres tarjetas o más. En una tarjeta AMD bajo Windows el instalador compila `strata-poly` para
la tarjeta, como en Linux (las wheels TheRock de AMD en `.rocm-win`), y avisa de que ejecutarlo allí es nuevo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

Desde la 0.1.44, el propio motor de [Strata](https://github.com/Niko1221/Strata) también ejecuta Qwen3.6, a partir de
UD-Q4_K_XL (`--family qwen36-strata`, 32 GB de RAM o más). En una RTX 5090 con UD-Q4_K_XL leyó un prompt de 4K 1,6
veces más rápido que strata-poly; con un prompt de 31K y al escribir quedaron casi iguales ([medido](../BACKENDS.md#qwen36-on-two-engines),
en inglés). Los tamaños más pequeños, para tarjetas de 8 GB, siguen en strata-poly.

## ¿Qué modelo elijo?

Tres tamaños por ahora. El instalador toma UD-Q3_K_M, o UD-Q2_K_XL en un PC con menos RAM. UD-Q4_K_M se elige por su nombre: `--model UD-Q4_K_M`.

| Tamaño | Qué es | Descarga | RAM junto a una tarjeta de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | el modelo entero con expertos de 3 bits ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 GB | unos 21 GB |
| **UD-Q2_K_XL** | el mismo con expertos de 2 y 3 bits: el que menos RAM pide | 13 GB | unos 17 GB |
| **UD-Q4_K_M** | el mismo con expertos de 4 bits | 23 GB | unos 27 GB |

Una tarjeta más grande guarda más parte del modelo y necesita menos RAM: con UD-Q3_K_M, unos 17 GB junto a una tarjeta de 12 GB, 13 GB junto a una de 16 GB y 10 GB cuando la tarjeta lo guarda todo. Son estimaciones del instalador. No se
han comprobado en un PC así.

## ¿Cómo funciona?

El diagrama y cómo funciona: [HOW_IT_WORKS_MODELS.es.md](../HOW_IT_WORKS_MODELS.es.md#qwen36-35b-a3b).

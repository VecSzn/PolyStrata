# ¿Cómo funciona?

Volver a [README.es.md](../README.es.md)

Los modelos como este suelen funcionar en servidores con cientos de gigabytes de memoria gráfica. Tu tarjeta gráfica
tiene 8-32 GB. PolyStrata hace que el modelo quepa **repartiendo el trabajo por todo tu PC**. Piensa en una cocina: lo
que usas todo el rato se queda en la encimera, y el resto espera en la despensa.

## Qwen3.8-Flash-Next

<p align="center"><img src="media/how-it-works.svg" width="860" alt="Los 24,576 expertos del modelo: los más usados en la tarjeta gráfica, todos en la RAM y una tabla de consulta en el SSD"></p>

- **El modelo es un equipo de 24,576 pequeños especialistas ("expertos").** Cada palabra solo necesita 10 de ellos.
- **Tu tarjeta gráfica** guarda los pocos miles de expertos que más se usan. **Tu RAM** los tiene todos, y **tu
  procesador** trabaja con el resto al mismo tiempo. **Tu SSD** guarda una gran tabla de consulta.

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="Un pequeño ayudante adivina las siguientes palabras; el modelo grande las comprueba todas a la vez y se queda con las correctas"></p>

- **Adivinar y comprobar:** un pequeño ayudante adivina las siguientes palabras. El modelo grande las comprueba todas
  a la vez. Recibes la misma respuesta, 1.6-1.8x antes.
- **Los textos largos se leen en trozos grandes** (hasta 8,192 tokens cada vez), a más de 1,000 tokens por segundo.

La explicación larga: [docs/HOW_IT_WORKS.md](HOW_IT_WORKS.md). Cada parte y sus números:
[los detalles](DETAILS.md#how-it-works) y el [artículo](paper/Strata-Paper.pdf).

## GLM 5.3 Flash

<p align="center"><img src="media/how-it-works-glm.svg" width="860" alt="Los 12,096 expertos del modelo: los más usados en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **El modelo es un equipo de 12,096 pequeños expertos**, 288 en cada una de 42 capas. Cada palabra solo necesita 8 de
  cada capa.
- **Tu tarjeta gráfica** guarda los más usados, unos 2,470 en una tarjeta de 32 GB: el motor cuenta cuáles usa tu
  trabajo y vuelve a elegir en el siguiente arranque. **Tu RAM** tiene los demás, y **tu procesador** los calcula al
  mismo tiempo. **Tu SSD** guarda el archivo del modelo, que se usa donde está, sin una segunda copia; un arranque
  tarda unos 6 segundos cuando el archivo ya está en tu RAM.

<p align="center"><img src="media/guess-and-check-glm.svg" width="860" alt="Un bloque del modelo adivina las siguientes palabras; el modelo las comprueba todas a la vez y se queda con las correctas"></p>

- **Adivinar y comprobar:** un bloque del modelo hecho para adivinar propone las siguientes palabras. El modelo las
  comprueba todas a la vez. Recibes la misma respuesta, 1.1-1.2x antes.
- **Los textos largos se leen en trozos grandes** (1,024 tokens cada vez; los expertos que la tarjeta no tiene pasan
  por ella), a entre 780 y 1,250 tokens por segundo.

Cada parte y sus cifras: [docs/GLM.md](GLM.md#how-it-uses-the-card) (en inglés).

## DeepSeek V4 Flash

<p align="center"><img src="media/how-it-works-dsv4.svg" width="860" alt="Los 11,008 expertos del modelo: los más usados en la tarjeta gráfica, el resto en la RAM, los archivos del modelo en el SSD"></p>

- **El modelo es un equipo de 11,008 pequeños expertos**, 256 en cada una de sus 43 capas. Cada palabra necesita solo
  6 de cada capa, además de uno que usan todas las palabras.
- **Tu tarjeta gráfica** guarda los más usados, unos 2,840 en una tarjeta de 32 GB: el motor cuenta cuáles usa tu
  trabajo. **Tu RAM** guarda el resto, y **tu procesador** los calcula a la vez. **Tu SSD** guarda los archivos del
  modelo, que se usan donde están, sin una segunda copia.
- **Una memoria pequeña de la conversación:** una palabra lee enteras las 128 últimas palabras y el resto como filas
  comprimidas, una por cada 4 o por cada 128 palabras. Un token de contexto ocupa 6,880 bytes, así que un contexto
  largo deja a la tarjeta casi todo su espacio para los expertos.
- **Los textos largos se leen en trozos grandes** (1,024 tokens a la vez), 30,000 tokens a unos 1,000 tokens por
  segundo. Todavía no tiene "adivinar y comprobar": el modelo borrador hecho para él aún no funciona.

Cada parte y sus cifras: [docs/DEEPSEEK.md](DEEPSEEK.md#how-it-differs-from-the-other-models) (en inglés).

## LFM2.5-24B-A2B

<p align="center"><img src="media/how-it-works-lfm25.svg" width="860" alt="Los 2,432 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **El modelo es un equipo de 2,432 pequeños expertos**, 64 en cada una de 38 capas. Cada palabra solo necesita 4 de
  cada capa.
- **La mayoría de sus 40 capas no tiene atención.** 30 de ellas mezclan las palabras con una convolución corta sobre
  las tres últimas y solo guardan dos vectores por capa. Solo las otras 10 recuerdan toda la conversación, así que su
  memoria de una conversación es pequeña: un contexto de 128K solo ocupa 2 GB de VRAM más que 32K.
- **Tu tarjeta gráfica** guarda todos los expertos que caben: unos 900 a 1,300 en una tarjeta de 8 GB, y a partir de
  24 GB caben todos en los tamaños de 4 bits. En la tarjeta de 8 GB los que tiene sirven de dos tercios a cuatro
  quintos de lo que usa una respuesta. **Tu RAM** tiene los demás, y **tu procesador** los calcula al mismo tiempo.
  **Tu SSD** guarda el archivo del modelo, que se usa donde está, sin una segunda copia.
- **Sin bloque de borrador.** El modelo no tiene nada que adivine lo siguiente, así que el motor solo adivina a
  partir del texto que la conversación repite.
- **Cuando la tarjeta guarda todos los expertos**, el paso entero se queda en la tarjeta: ella misma elige los
  expertos de las palabras y ejecuta las 40 capas de un token como un solo lanzamiento grabado.

Cada parte y sus números: [docs/LFM.md](LFM.md) (en inglés).

## Flint 40B-A2.5B

<p align="center"><img src="media/how-it-works-flint.svg" width="860" alt="Los 12,032 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **El modelo es un equipo de 12,032 pequeños expertos**, 256 en cada una de 47 capas. Cada palabra solo necesita 6 de
  cada capa, y un experto compartido que usa cada palabra.
- **La mayoría de sus 49 capas mira 513 tokens atrás.** 35 de ellas solo atienden a los últimos 513 tokens y guardan
  un anillo de filas que no crece. Solo las otras 14 recuerdan toda la conversación, así que su memoria de una
  conversación es pequeña: un contexto de 128K solo ocupa 2,6 GB de VRAM más que 32K.
- **Tu tarjeta gráfica** guarda todos los expertos que caben: unos 1,600 a 2,500 en una tarjeta de 8 GB, todos los de
  UD-Q4_K_XL en una de 32 GB. **Tu RAM** tiene los demás, y **tu procesador** los calcula al mismo tiempo. **Tu SSD**
  guarda el archivo del modelo, que se usa donde está, sin una segunda copia.
- **Sin bloque de borrador.** El modelo no tiene nada que adivine lo siguiente, así que el motor solo adivina a
  partir del texto que la conversación repite.

Cada parte y sus números: [docs/FLINT.md](FLINT.md) (en inglés).

## Qwen3.6-35B-A3B

<p align="center"><img src="media/how-it-works-qwen36.svg" width="860" alt="Los 10,240 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **El modelo es un equipo de 10,240 pequeños expertos**, 256 en cada una de 40 capas. Cada palabra solo necesita 8 de
  cada capa.
- **Tu tarjeta gráfica** guarda todos los que caben: unos 2,600 en una tarjeta de 8 GB, y a partir de 24 GB caben
  todos. Mientras escribe, el motor trae a la tarjeta los expertos que la respuesta elige una y otra vez, y en la
  tarjeta de 8 GB los que tiene sirven alrededor del 60 % de lo que usa una respuesta. **Tu RAM** tiene los demás, y
  **tu procesador** los calcula al mismo tiempo. **Tu SSD** guarda el archivo del modelo, que se usa donde está, sin
  una segunda copia.

<p align="center"><img src="media/guess-and-check-qwen36.svg" width="860" alt="El bloque borrador del modelo adivina las siguientes palabras; el modelo las comprueba todas a la vez y se queda con las correctas"></p>

- **Adivinar y comprobar:** el bloque borrador del propio modelo, que está en su archivo, propone las siguientes
  palabras. El modelo las comprueba todas a la vez. Recibes la misma respuesta, 1.2-1.5x antes.
- **Los textos largos se leen en trozos grandes** (1,024 tokens cada vez). Cuando la tarjeta no tiene todos los
  expertos, los demás pasan por ella una sola vez para todo el prompt.

Cada parte y sus cifras: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en inglés).

## Clover-35B-A3B-Preview

<p align="center"><img src="media/how-it-works-clover.svg" width="860" alt="Los 10,240 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **Qwen3.6-35B-A3B entrenado más:** un equipo de 10,240 pequeños expertos, 256 en cada una de 40 capas, y solo 8 de
  cada capa para cada palabra.
- **Tu tarjeta gráfica** guarda todos los que caben: unos 2,200 a 4,100 en una tarjeta de 8 GB, más cuanto más pequeño
  es el tamaño, y a partir de 24 GB todos los de Q3_K_M. En la tarjeta de 8 GB los que tiene sirven del 60 al 80 % de lo
  que usa una respuesta. **Tu RAM** tiene los demás, y **tu procesador** los calcula al mismo tiempo. **Tu SSD** guarda
  el archivo del modelo, que se usa donde está, sin una segunda copia.
- **Sin bloque de borrador.** Clover se entrenó para escribir un bloque de palabras de una vez, rellenando huecos
  juntos, y así también puede adivinar las palabras siguientes. Pero cada bloque adivinado vuelve a ejecutar el modelo
  entero, y eso costó más de lo que ahorraron los aciertos, así que está apagado
  ([medido](BACKENDS.md#clover-35b-a3b-preview), en inglés). El motor solo adivina a partir del texto que la
  conversación repite.
- **Los textos largos se leen en trozos grandes** (1,024 tokens cada vez), como con Qwen3.6-35B-A3B.

Cada parte y sus cifras: [docs/BACKENDS.md](BACKENDS.md#clover-35b-a3b-preview) (en inglés).

## Ornith 1.5

<p align="center"><img src="media/how-it-works-ornith.svg" width="860" alt="Los 10,240 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **La arquitectura de Qwen3.6-35B-A3B con otros pesos:** un equipo de 10,240 pequeños expertos, 256 en cada una de 40
  capas, y solo 8 de cada capa para cada palabra.
- **Tu tarjeta gráfica** guarda todos los que caben: unos 3,100 en una tarjeta de 8 GB, y a partir de 24 GB caben
  todos. Mientras escribe, el motor trae a la tarjeta los expertos que la respuesta elige una y otra vez, y en la
  tarjeta de 8 GB los que tiene sirven alrededor del 60 % de lo que usa una respuesta. **Tu RAM** tiene los demás, y
  **tu procesador** los calcula al mismo tiempo. **Tu SSD** guarda el archivo del modelo, que se usa donde está, sin
  una segunda copia.

<p align="center"><img src="media/guess-and-check-ornith.svg" width="860" alt="El bloque borrador del modelo adivina las siguientes palabras; el modelo las comprueba todas a la vez y se queda con las correctas"></p>

- **Adivinar y comprobar:** su bloque borrador también está en su archivo y propone las siguientes palabras. Acierta
  menos veces que el de Qwen3.6-35B-A3B, así que recibes la misma respuesta 1.1-1.2x antes.
- **Los textos largos se leen en trozos grandes** (1,024 tokens cada vez). Cuando la tarjeta no tiene todos los
  expertos, los demás pasan por ella una sola vez para todo el prompt.

Cada parte y sus cifras: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en inglés).

## Gemma 4 26B-A4B

<p align="center"><img src="media/how-it-works-gemma4.svg" width="860" alt="Los 3,840 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **El modelo es un equipo de 3,840 pequeños expertos**, 128 en cada una de 30 capas. Cada palabra solo necesita 8 de
  cada capa, además de una parte de la capa que usan todas las palabras.
- **Tu tarjeta gráfica** guarda todos los que caben: unos 1,000 en una tarjeta de 8 GB, y según la estimación del
  instalador a partir de 16 GB caben todos. Mientras escribe, el motor trae a la tarjeta los expertos que la respuesta
  elige una y otra vez, y en la tarjeta de 8 GB los que tiene sirven alrededor del 55 % de lo que usa una respuesta.
  **Tu RAM** tiene los demás, y **tu procesador** los calcula al mismo tiempo. **Tu SSD** guarda el archivo del modelo
  y su modelo borrador, que se usan donde están, sin una segunda copia.

<p align="center"><img src="media/guess-and-check-gemma4.svg" width="860" alt="El pequeño modelo borrador de Google adivina las siguientes palabras; el modelo las comprueba todas a la vez y se queda con las correctas"></p>

- **Adivinar y comprobar:** el pequeño modelo borrador de Google para él, un archivo de 0.5 GB que el instalador
  descarga con el modelo, propone las siguientes palabras. El modelo las comprueba todas a la vez. Recibes la misma
  respuesta, 1.3-1.6x antes.
- **Los textos largos se leen en trozos grandes** (1,024 tokens cada vez). Cuando la tarjeta no tiene todos los
  expertos, los demás pasan por ella una sola vez para todo el prompt.

Cada parte y sus cifras: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en inglés).

## Tiel Coder 35B-A3B

<p align="center"><img src="media/how-it-works-tiel.svg" width="860" alt="Los 10,240 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **Ornith 1.5 con otra cuantización:** un equipo de 10,240 pequeños expertos, 256 en cada una de 40
  capas, y solo 8 de cada capa para cada palabra.
- **Tu tarjeta gráfica** guarda todos los que caben: unos 2,700 en una tarjeta de 8 GB, y a partir de 24 GB caben
  todos. Mientras escribe, el motor trae a la tarjeta los expertos que la respuesta elige una y otra vez, y en la
  tarjeta de 8 GB los que tiene sirven alrededor del 50 % de lo que usa una respuesta. **Tu RAM** tiene los demás, y
  **tu procesador** los calcula al mismo tiempo. **Tu SSD** guarda el archivo del modelo, que se usa donde está, sin
  una segunda copia.

<p align="center"><img src="media/guess-and-check-tiel.svg" width="860" alt="El bloque borrador del modelo adivina las siguientes palabras; el modelo las comprueba todas a la vez y se queda con las correctas"></p>

- **Adivinar y comprobar:** su bloque borrador también está en su archivo y propone las siguientes palabras. Acierta
  menos veces que el de Qwen3.6-35B-A3B, así que recibes la misma respuesta 1.2-1.3x antes.
- **Los textos largos se leen en trozos grandes** (1,024 tokens cada vez). Cuando la tarjeta no tiene todos los
  expertos, los demás pasan por ella una sola vez para todo el prompt.

Cada parte y sus cifras: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en inglés).

## MiniCPM-V 4.7 35B-A3B

<p align="center"><img src="media/how-it-works-minicpmv.svg" width="860" alt="Los 10,240 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **La arquitectura de Qwen3.6-35B-A3B con los pesos de OpenBMB:** un equipo de 10,240 pequeños expertos, 256 en cada
  una de 40 capas, y solo 8 de cada capa para cada palabra.
- **Tu tarjeta gráfica** guarda todos los que caben: con Q4_K_M unos 2,000 en una tarjeta de 8 GB, y en una de 32 GB
  caben todos. Mientras escribe, el motor trae a la tarjeta los expertos que la respuesta elige una y otra vez, y en la
  tarjeta de 8 GB los que tiene sirven alrededor del 60 al 70 % de lo que usa una respuesta. **Tu RAM** tiene los
  demás, y **tu procesador** los calcula al mismo tiempo. **Tu SSD** guarda el archivo del modelo, que se usa donde
  está, sin una segunda copia.
- **Una palabra cada vez:** su archivo no tiene bloque borrador, así que nada adivina por adelantado.
- **Imágenes:** su propio codificador de imágenes lee una imagen grande como una vista general y unos cuantos
  mosaicos. Una captura de 1,200 por 398 píxeles se convierte en unos 300 tokens, un gráfico de 2,520 por 1,620 en
  unos 600.
- **Los textos largos se leen en trozos grandes** (1,024 tokens cada vez). Cuando la tarjeta no tiene todos los
  expertos, los demás pasan por ella una sola vez para todo el prompt.

Cada parte y sus cifras: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en inglés).

## Kolibri 1

<p align="center"><img src="media/how-it-works-kolibri.svg" width="860" alt="Los 19,200 expertos del modelo: los que una respuesta elige una y otra vez en la tarjeta gráfica, los demás en la RAM y el archivo del modelo en el SSD"></p>

- **El modelo es un equipo de 19,200 pequeños expertos**, 384 en cada una de 50 capas. Cada palabra solo necesita 6 de
  cada capa, y un experto compartido que usa cada palabra.
- **Cuatro de cada cinco capas miran 513 tokens atrás.** Guardan un anillo de filas que no crece. La quinta recuerda
  toda la conversación y no usa ninguna posición de las palabras.
- **Tu tarjeta gráfica** guarda todos los expertos que caben: alrededor de una décima parte en una tarjeta de 8 GB,
  todos los de Q2_K en una de 32 GB. **Tu RAM** tiene los demás, y **tu procesador** los calcula al mismo tiempo. **Tu
  SSD** guarda el archivo del modelo, que se usa donde está, sin una segunda copia.
- **Sin bloque de borrador.** El modelo no tiene nada que adivine lo siguiente, así que el motor solo adivina a
  partir del texto que la conversación repite.

Cada parte y sus cifras: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (en inglés).

# How does it work?

Back to [README.md](../README.md)

Models like these usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
8-32 GB. PolyStrata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

## Qwen3.8-Flash-Next

<p align="center"><img src="media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](HOW_IT_WORKS.md). Every part and its numbers:
[the details](DETAILS.md#how-it-works) and the [paper](paper/Strata-Paper.pdf).

## GLM 5.3 Flash

<p align="center"><img src="media/how-it-works-glm.svg" width="860" alt="The model's 12,096 experts: the most used on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 12,096 small experts**, 288 in each of 42 layers. Each word needs only 8 from every layer.
- **Your graphics card** keeps the ones used most often, about 2,470 of them on a 32 GB card: the engine counts which
  ones your work uses and chooses again at the next start. **Your RAM** holds the rest, and **your processor** runs
  them at the same time. **Your SSD** holds the model's file, which is used where it is, with no second copy; a start
  takes about 6 seconds once the file is in your RAM.

<p align="center"><img src="media/guess-and-check-glm.svg" width="860" alt="A block of the model guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a block of the model made for guessing proposes the next few words. The model checks them all
  at once. You get the same answer, 1.1-1.2x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time, the experts the card does not hold passing through
  it), at 780 to 1,250 tokens per second.

Every part and its numbers: [docs/GLM.md](GLM.md#how-it-uses-the-card).

## DeepSeek V4 Flash

<p align="center"><img src="media/how-it-works-dsv4.svg" width="860" alt="The model's 11,008 experts: the most used on the graphics card, the rest in RAM, the model's files on the SSD"></p>

- **The model is a team of 11,008 small experts**, 256 in each of 43 layers. Each word needs only 6 from every
  layer, beside one that every word uses.
- **Your graphics card** keeps the ones used most often, about 2,840 of them on a 32 GB card: the engine counts
  which ones your work uses. **Your RAM** holds the rest, and **your processor** runs them at the same time. **Your
  SSD** holds the model's files, which are used where they are, with no second copy.
- **A small memory of the conversation:** a word reads the last 128 words in full and the rest as compressed rows,
  one for every 4 or every 128 words. A token of context takes 6,880 bytes, so a long context leaves the card
  nearly all of its room for experts.
- **Long texts are read in big pieces** (1,024 tokens at a time), 30,000 tokens at about 1,000 tokens per second.
  It has no guess-then-check yet: the draft model made for it is not run.

Every part and its numbers: [docs/DEEPSEEK.md](DEEPSEEK.md#how-it-differs-from-the-other-models).

## LFM2.5-24B-A2B

<p align="center"><img src="media/how-it-works-lfm25.svg" width="860" alt="The model's 2,432 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 2,432 small experts**, 64 in each of 38 layers. Each word needs only 4 from every layer.
- **Most of its 40 layers have no attention.** 30 of them mix the words with a short convolution over the last three,
  which keeps two vectors a layer. Only the other 10 keep the whole conversation, so its memory of a conversation is
  small: a 128K context takes 2 GB more VRAM than 32K.
- **Your graphics card** keeps as many experts as fit: about 900 to 1,300 of them on an 8 GB card, all of them from
  24 GB on for the 4-bit sizes. On the 8 GB card the ones it holds serve two thirds to four fifths of what an answer
  uses. **Your RAM** holds the rest, and **your processor** runs them at the same time. **Your SSD** holds the
  model's file, which is used where it is, with no second copy.
- **No draft block.** The model has nothing that guesses ahead, so the engine guesses only from text the
  conversation repeats.
- **When the card holds every expert**, the whole step stays on the card: it routes the words to their experts and
  runs a token's 40 layers as one recorded launch.

Every part and its numbers: [docs/LFM.md](LFM.md).

## Flint 40B-A2.5B

<p align="center"><img src="media/how-it-works-flint.svg" width="860" alt="The model's 12,032 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 12,032 small experts**, 256 in each of 47 layers. Each word needs only 6 from every layer,
  and one shared expert that every word uses.
- **Most of its 49 layers look back 513 tokens.** 35 of them attend only to the last 513 tokens and keep a ring of
  rows that does not grow. Only the other 14 keep the whole conversation, so its memory of a conversation is small:
  a 128K context takes 2.6 GB more VRAM than 32K.
- **Your graphics card** keeps as many experts as fit: about 1,600 to 2,500 of them on an 8 GB card, all of UD-Q4_K_XL
  on a 32 GB card. **Your RAM** holds the rest, and **your processor** runs them at the same time. **Your SSD** holds
  the model's file, which is used where it is, with no second copy.
- **No draft block.** The model has nothing that guesses ahead, so the engine guesses only from text the
  conversation repeats.

Every part and its numbers: [docs/FLINT.md](FLINT.md).

## Qwen3.6-35B-A3B

<p align="center"><img src="media/how-it-works-qwen36.svg" width="860" alt="The model's 10,240 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 10,240 small experts**, 256 in each of 40 layers. Each word needs only 8 from every layer.
- **Your graphics card** keeps as many as fit: about 2,600 of them on an 8 GB card, all of them from 24 GB on. While
  it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB card the ones it holds serve
  about 60% of what an answer uses. **Your RAM** holds the rest, and **your processor** runs them at the same time.
  **Your SSD** holds the model's file, which is used where it is, with no second copy.

<p align="center"><img src="media/guess-and-check-qwen36.svg" width="860" alt="The model's draft block guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** the model's own draft block, which is in its file, proposes the next few words. The model
  checks them all at once. You get the same answer, 1.2-1.5x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works).

## Clover-35B-A3B-Preview

<p align="center"><img src="media/how-it-works-clover.svg" width="860" alt="The model's 10,240 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **Qwen3.6-35B-A3B trained further:** a team of 10,240 small experts, 256 in each of 40 layers, and only 8 of every
  layer for each word.
- **Your graphics card** keeps as many as fit: about 2,200 to 4,100 of them on an 8 GB card, the more the smaller the
  size, and all of Q3_K_M from 24 GB on. On the 8 GB card the ones it holds serve 60 to 80% of what an answer uses.
  **Your RAM** holds the rest, and **your processor** runs them at the same time. **Your SSD** holds the model's file,
  which is used where it is, with no second copy.
- **No draft block.** Clover was trained to write a block of words at once, filling in blanks together, and it can
  guess the next words that way. That runs the whole model once more for each block of guesses, which cost more than
  the right guesses saved, so it is off ([measured](BACKENDS.md#clover-35b-a3b-preview)). The engine guesses only from
  text the conversation repeats.
- **Long texts are read in big pieces** (1,024 tokens at a time), as with Qwen3.6-35B-A3B.

Every part and its numbers: [docs/BACKENDS.md](BACKENDS.md#clover-35b-a3b-preview).

## Ornith 1.5

<p align="center"><img src="media/how-it-works-ornith.svg" width="860" alt="The model's 10,240 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **Qwen3.6-35B-A3B's architecture with other weights:** a team of 10,240 small experts, 256 in each of 40 layers, and
  only 8 of every layer for each word.
- **Your graphics card** keeps as many as fit: about 3,100 of them on an 8 GB card, all of them from 24 GB on. While
  it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB card the ones it holds serve
  about 60% of what an answer uses. **Your RAM** holds the rest, and **your processor** runs them at the same time.
  **Your SSD** holds the model's file, which is used where it is, with no second copy.

<p align="center"><img src="media/guess-and-check-ornith.svg" width="860" alt="The model's draft block guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** its draft block is in its file too, and proposes the next few words. It guesses right less
  often than Qwen3.6-35B-A3B's, so you get the same answer 1.1-1.2x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works).

## Gemma 4 26B-A4B

<p align="center"><img src="media/how-it-works-gemma4.svg" width="860" alt="The model's 3,840 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 3,840 small experts**, 128 in each of 30 layers. Each word needs only 8 from every layer,
  beside a part of the layer that every word uses.
- **Your graphics card** keeps as many as fit: about 1,000 of them on an 8 GB card, all of them from 16 GB on by the
  installer's estimate. While it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB
  card the ones it holds serve about 55% of what an answer uses. **Your RAM** holds the rest, and **your processor**
  runs them at the same time. **Your SSD** holds the model's file and its draft model, which are used where they are,
  with no second copy.

<p align="center"><img src="media/guess-and-check-gemma4.svg" width="860" alt="Google's small draft model guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** Google's small draft model for it, a file of 0.5 GB that the installer downloads with the
  model, proposes the next few words. The model checks them all at once. You get the same answer, 1.3-1.6x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works).

## Tiel Coder 35B-A3B

<p align="center"><img src="media/how-it-works-tiel.svg" width="860" alt="The model's 10,240 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **Ornith 1.5 in another quantization:** a team of 10,240 small experts, 256 in each of 40 layers, and
  only 8 of every layer for each word.
- **Your graphics card** keeps as many as fit: about 2,700 of them on an 8 GB card, all of them from 24 GB on. While
  it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB card the ones it holds serve
  about 50% of what an answer uses. **Your RAM** holds the rest, and **your processor** runs them at the same time.
  **Your SSD** holds the model's file, which is used where it is, with no second copy.

<p align="center"><img src="media/guess-and-check-tiel.svg" width="860" alt="The model's draft block guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** its draft block is in its file too, and proposes the next few words. It guesses right less
  often than Qwen3.6-35B-A3B's, so you get the same answer 1.2-1.3x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works).

## MiniCPM-V 4.7 35B-A3B

<p align="center"><img src="media/how-it-works-minicpmv.svg" width="860" alt="The model's 10,240 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **Qwen3.6-35B-A3B's architecture with OpenBMB's weights:** a team of 10,240 small experts, 256 in each of 40
  layers, and only 8 of every layer for each word.
- **Your graphics card** keeps as many as fit: about 2,000 of them on an 8 GB card with Q4_K_M, all of them on a 32 GB
  card. While it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB card the ones it
  holds serve about 60 to 70% of what an answer uses. **Your RAM** holds the rest, and **your processor** runs them at
  the same time. **Your SSD** holds the model's file, which is used where it is, with no second copy.
- **One word at a time:** its file has no draft block, so nothing guesses ahead.
- **Pictures:** its own picture encoder reads a large picture as an overview and a few tiles. A screenshot of 1,200 by
  398 pixels becomes about 300 tokens, a chart of 2,520 by 1,620 about 600.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works).

## Kolibri 1

<p align="center"><img src="media/how-it-works-kolibri.svg" width="860" alt="The model's 19,200 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 19,200 small experts**, 384 in each of 50 layers. Each word needs only 6 from every layer,
  and one shared expert that every word uses.
- **Four layers in five look back 513 tokens.** They keep a ring of rows that does not grow. The fifth keeps the whole
  conversation and has no word positions at all.
- **Your graphics card** keeps as many experts as fit: about a tenth of them on an 8 GB card, all of Q2_K on a 32 GB
  card. **Your RAM** holds the rest, and **your processor** runs them at the same time. **Your SSD** holds the model's
  file, which is used where it is, with no second copy.
- **No draft block.** The model has nothing that guesses ahead, so the engine guesses only from text the
  conversation repeats.

Every part and its numbers: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works).

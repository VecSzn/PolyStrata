# LFM2.5-24B-A2B

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**LFM2.5-24B-A2B** da Liquid AI, um arquivo de 11 a 25 GB. A maioria das camadas dele não tem atenção, então uma conversa longa ocupa pouca VRAM: um contexto de 128K ocupa só 2 GB a mais que 32K. A licença dele só permite o uso comercial a empresas que faturam menos de 10 milhões de dólares por ano.

## Qual é a velocidade?

Medido como o [Qwen3.6-35B-A3B](qwen36.pt-BR.md), nos mesmos dois PCs, com um prompt de cerca de 30,000 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 70-73 tokens/s | 3,590-3,620 tokens/s |
| **UD-Q3_K_XL** | 88 tokens/s | 3,760-3,810 tokens/s |
| **QAD-Q4_0** | 77-78 tokens/s | 3,720-3,730 tokens/s |
| **UD-Q6_K_XL** | 42-43 tokens/s | 3,030-3,120 tokens/s |
| **Q8_0** | 23-34 tokens/s | 2,950-3,040 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 480-485 tokens/s | 23,390-23,460 tokens/s |
| **UD-Q3_K_XL** | 490-497 tokens/s | 22,920-23,400 tokens/s |
| **QAD-Q4_0** | 498 tokens/s | 23,880-23,900 tokens/s |
| **UD-Q6_K_XL** | 423-426 tokens/s | 21,250-21,270 tokens/s |
| **Q8_0** | 376 tokens/s | 23,270-23,350 tokens/s |

</td></tr>
</table>

No notebook o Q8_0 enche os 32 GB de RAM, e por isso é lento lá; a estação de trabalho guarda inteiro cada tamanho.

No notebook o llama.cpp b11540 com `--n-cpu-moe 25`, o melhor dos ajustes testados lá, leu um prompt de 32,000 tokens
a 530 tokens/s e escreveu depois a 53; o PolyStrata o leu a 3,540 e escreveu a 71-73. Na estação de trabalho o
llama-bench leu 29,816 tokens a 13,300 tokens/s com o lote padrão do llama.cpp, de 512, e a 23,600 com 2,048, tão
rápido quanto o PolyStrata, e escreveu token a token a 455 tokens/s, e a 369 depois desse prompt; o PolyStrata escreve
a 452 e 387. Um prompt de cerca de 124,000 tokens em um contexto de 128K foi lido e respondido em 8,7 segundos na
estação de trabalho; o notebook o leu em 45 segundos.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 4 GB são ocupados antes do primeiro expert. Ele rodou na placa de 8 GB de um notebook e numa placa de 32 GB. |
| **RAM** | Cerca de **16, 18, 20, 27 ou 31 GB** ao lado de uma placa de 8 GB para UD-Q3_K_XL, QAD-Q4_0, UD-Q4_K_XL, UD-Q6_K_XL e Q8_0, pela estimativa do instalador. Uma placa de 24 GB guarda inteiros os três tamanhos menores. Os cinco rodaram em um notebook com 32 GB. |
| **Disco** | 11 a 25 GB para o arquivo do modelo. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como com o [Qwen3.6-35B-A3B](qwen36.pt-BR.md). Um usuário já o rodou numa AMD RX 6700 XT (12 GB) no Linux. Ele
só lê texto. A licença da Liquid AI só permite o uso comercial a empresas que faturam menos de 10 milhões de dólares
por ano ([LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)).

A lista completa: [docs/LFM.md](../LFM.md#what-you-need) (em inglês).

## Qual modelo devo escolher?

Cinco tamanhos. O instalador escolhe UD-Q4_K_XL, ou UD-Q3_K_XL num PC com menos RAM. Os outros são escolhidos pelo nome: `--model QAD-Q4_0`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | o modelo inteiro com experts de 4 bits ([Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)) | 15 GB | cerca de 20 GB |
| **UD-Q3_K_XL** | o mesmo com experts de 3 bits: o que pede menos RAM | 11 GB | cerca de 16 GB |
| **QAD-Q4_0** | 4 bits, da própria [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF), treinado para perder menos com essa quantização | 14 GB | cerca de 18 GB |
| **UD-Q6_K_XL** | experts de 6 bits | 22 GB | cerca de 27 GB |
| **Q8_0** | 8 bits em tudo, o mais próximo do modelo como foi treinado | 25 GB | cerca de 31 GB |

Contra o Q8_0, em cinco prompts de 256 tokens cada, os outros tamanhos escolhem o token mais provável do Q8_0 em 96,8%
dos passos (UD-Q6_K_XL), 95,5% (UD-Q4_K_XL), 92,7% (QAD-Q4_0) e 91,6% (UD-Q3_K_XL).

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#lfm25-24b-a2b).

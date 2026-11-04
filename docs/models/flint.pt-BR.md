# Flint 40B-A2.5B

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Flint 40B-A2.5B** da Recursive, feito para agentes de código: um arquivo de 18 a 42 GB. A maioria das camadas dele só olha 513 tokens para trás, então uma conversa longa ocupa pouca VRAM: um contexto de 128K ocupa só 2,6 GB a mais que 32K.

## Qual é a velocidade?

Medido como o [LFM2.5-24B-A2B](lfm25.pt-BR.md), nos mesmos dois PCs, com um prompt de cerca de 29,000 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 90 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 20-29 tokens/s | 1,310-1,420 tokens/s |
| **UD-Q3_K_XL** | 61-63 tokens/s | 2,000-2,010 tokens/s |
| **UD-Q6_K_XL** | Não cabe | Não cabe |
| **Q8_0** | Não cabe | Não cabe |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 251-252 tokens/s | 12,900 tokens/s |
| **UD-Q3_K_XL** | 260-264 tokens/s | 13,080-13,200 tokens/s |
| **Q8_0** | 158-160 tokens/s | 6,740-6,970 tokens/s |

</td></tr>
</table>

No notebook só o UD-Q3_K_XL fica nos 32 GB de RAM ao lado dos programas da área de trabalho. Os tamanhos maiores leem
os experts do SSD de novo e de novo, e por isso são lentos lá; o instalador escolhe lá o UD-Q3_K_XL. A placa da
estação de trabalho guarda inteiros o UD-Q4_K_XL e o UD-Q3_K_XL; o UD-Q6_K_XL não rodou lá.

No notebook o llama.cpp com `--fit on`, o mais rápido dos ajustes testados lá, leu um prompt de 29,442 tokens a 390
tokens/s e escreveu depois a 35; o PolyStrata o leu a 1,950-2,010 e escreveu a 49-52
([as medições](../../bench/results/2026-10-24-laptop-flint-llamacpp-a37cc46/README.md)).

Na estação de trabalho o llama-bench do llama.cpp leu 29,816 tokens a 7,790 tokens/s com o lote padrão de 512 e a
12,470 com 2,048, o PolyStrata a 12,910-12,940; token a token o llama.cpp escreveu a 222 tokens/s e a 188 depois desse
prompt, o PolyStrata a 249-256 e 212-213. Um prompt de 124,122 tokens em um contexto de 128K foi lido e respondido em 20
segundos, e os três fatos escondidos nele foram encontrados.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 4 GB são ocupados antes do primeiro expert. Ele rodou na placa de 8 GB de um notebook e numa placa de 32 GB. |
| **RAM** | Cerca de **22, 29, 40 ou 46 GB** ao lado de uma placa de 8 GB para UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL e Q8_0, pela estimativa do instalador. Num notebook com 32 GB só o UD-Q3_K_XL ficou na RAM, então o instalador escolhe o UD-Q4_K_XL a partir de cerca de 40 GB. Uma placa de 32 GB guarda o UD-Q4_K_XL inteiro. |
| **Disco** | 18 a 42 GB para o arquivo do modelo. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como com o [Qwen3.6-35B-A3B](qwen36.pt-BR.md). Ainda não foi testado numa placa AMD. Ele
só lê texto. A licença dele é Apache 2.0.

A lista completa: [docs/FLINT.md](../FLINT.md#what-you-need) (em inglês).

## Qual modelo devo escolher?

Quatro tamanhos. Ao lado de uma placa de 8 GB o instalador escolhe o UD-Q4_K_XL a partir de cerca de 40 GB de RAM e o
UD-Q3_K_XL abaixo disso; ao lado de uma placa maior o UD-Q4_K_XL precisa de menos. Os outros são escolhidos pelo nome:
`--model Q8_0`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | o modelo inteiro com experts de 4 e 5 bits ([Unsloth](https://huggingface.co/unsloth/flint-GGUF)) | 25 GB | cerca de 29 GB |
| **UD-Q3_K_XL** | o mesmo com experts de 3 bits: a menor RAM | 18 GB | cerca de 22 GB |
| **UD-Q6_K_XL** | experts de 6 bits | 36 GB | cerca de 40 GB |
| **Q8_0** | 8 bits em tudo, o mais próximo do modelo como foi treinado | 42 GB | cerca de 46 GB |

No notebook de 32 GB o arquivo do UD-Q4_K_XL não ficou na RAM: ele escreveu um chat curto a 20-29 tokens/s, o
UD-Q3_K_XL a 61-63. Contra o Q8_0, em cinco prompts de 256 tokens cada, os outros tamanhos escolhem o token mais
provável do Q8_0 em 97,8% dos passos (UD-Q6_K_XL), 95,9% (UD-Q4_K_XL) e 94,0% (UD-Q3_K_XL).

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#flint-40b-a25b).

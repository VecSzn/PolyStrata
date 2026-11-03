# Kolibri 1

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Kolibri 1** da Aleph Alpha, um modelo de 19,200 experts que escreve alemão tão bem quanto inglês: um arquivo de 29 a
48 GB, que fica quase todo na RAM ao lado de uma placa pequena. Quatro de cada cinco camadas olham só 513 tokens para
trás.

## Qual é a velocidade?

Medido como o [Flint 40B-A2.5B](flint.pt-BR.md), com um prompt de 31,676 tokens.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q4_K_M** | Não medido | Não medido |
| **Q2_K** | Não medido | Não medido |
| **Q3_K_M** | Não medido | Não medido |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q2_K** | 189-192 tokens/s | 4,980-5,070 tokens/s |

</td></tr>
</table>

A placa da estação de trabalho guarda os 19,200 experts do Q2_K (25,6 GiB). Ele começou a responder o prompt de 31,676
tokens depois de 6,3 a 6,4 segundos e escreveu depois a 129-131 tokens/s. O Q4_K_M e o Q3_K_M não rodaram ali, e no
notebook nenhum tamanho rodou.

O pull request que adicionou o modelo o mediu em outros PCs. Numa RTX 3060 (12 GB) com 24 GB de RAM no Windows, o Q2_K
escreveu um chat curto a 21 tokens/s e leu um prompt de 32K a 698. Numa Tesla V100 (32 GB), o Q4_K_M escreveu a 91
tokens/s e leu um prompt de 32K a 1,017; com duas, leu a 2,179.

Contra o llama.cpp com o patch `kolibri1` da conversão, seguindo a resposta greedy dele a um prompt em inglês e outro em
alemão, o Q2_K escolhe o mesmo token mais provável em 120 e 126 de 128 passos. Onde eles diferem, o token do llama.cpp é
a segunda ou terceira opção deste motor em todos os passos menos um, quase tão provável quanto.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 4 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 12 GB, numa Tesla V100 de 32 GB e numa RTX 5090 de 32 GB. |
| **RAM** | Cerca de **52, 33 ou 42 GB** ao lado de uma placa de 8 GB para Q4_K_M, Q2_K e Q3_K_M, pela estimativa do instalador. Ao lado de uma placa de 32 GB, o Q2_K precisa de cerca de 10 GB e o Q4_K_M de cerca de 26 GB. |
| **Disco** | 29 a 48 GB para o arquivo do modelo. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador prepara todo o resto, como para o [Qwen3.6-35B-A3B](qwen36.pt-BR.md). Ainda não foi testado numa placa AMD.
Ele só lê texto e não tem modelo de rascunho, então escreve um token por vez. A licença é Apache 2.0.

## Qual modelo devo escolher?

Três tamanhos, da conversão da [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF). Ao lado de uma placa de
8 GB, o instalador escolhe o Q4_K_M a partir de cerca de 48 GB de RAM e o Q2_K abaixo disso; ao lado de uma placa maior,
os dois precisam de menos. O Q3_K_M é escolhido pelo nome: `--model Q3_K_M`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **Q4_K_M** | 4 bits, com projeções down de 6 bits em metade das camadas | 48 GB | cerca de 52 GB |
| **Q2_K** | 2 bits: a menor RAM | 29 GB | cerca de 33 GB |
| **Q3_K_M** | 3 bits | 38 GB | cerca de 42 GB |

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#kolibri-1).

# Clover-35B-A3B-Preview

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Clover-35B-A3B-Preview** da DiffuSpace, um arquivo de 13 a 21 GB. É o Qwen3.6-35B-A3B treinado mais para escrever
também um bloco de tokens de uma vez. Ele só lê texto.

## Qual é a velocidade?

Medido pelo PolyStrata no notebook, com uma conversa curta e um prompt de 32,000 tokens. A placa dele tem 8 GB, e o
processador calcula os experts que não cabem na placa.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q3_K_M** | 68-73 tokens/s | 1,560-1,700 tokens/s |
| **APEX-I-NANOPLUS** | 72-76 tokens/s | 1,530-1,690 tokens/s |
| **Q4_K_M** | 38-58 tokens/s | 1,220-1,360 tokens/s |

</td></tr>
</table>

Depois do prompt de 32,000 tokens, o notebook escreve a 60-65 tokens/s com Q3_K_M. O Q4_K_M enche ali os 32 GB de RAM,
e é isso que o deixa lento e irregular. A estação de trabalho com a RTX 5090 não foi medida.

O Qwen3.6-35B-A3B escreve mais rápido no mesmo notebook, 94 a 100 tokens/s, porque o arquivo dele tem um bloco de
rascunho que adivinha as próximas palavras com o trabalho de uma camada. Os arquivos do Clover não têm. O Clover pode
adivinhar ele mesmo um bloco de palavras, mas isso roda o modelo inteiro de novo, e o PolyStrata deixa desligado: numa
placa que guardava todos os experts, as respostas chegaram antes sem isso
([medido](../BACKENDS.md#clover-35b-a3b-preview), em inglês).

Numa RTX PRO 6000 que guardava todos os experts do Q3_K_M, o PolyStrata escolheu o mesmo próximo token que o llama.cpp
(o suporte `clover_moe` do pull request 31306 dele) em 127 de 128 passos depois de um prompt curto, e em 125 de 128
depois de 5,573 tokens.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 3,5 GB são ocupados antes do primeiro expert. Ele rodou na placa de 8 GB de um notebook e numa placa de 96 GB. Ainda não foi testado numa placa AMD. |
| **RAM** | Cerca de **16, 20 ou 25 GB** ao lado de uma placa de 8 GB para APEX-I-NANOPLUS, Q3_K_M e Q4_K_M, pela estimativa do instalador. Uma placa de 24 GB guarda inteiros os dois tamanhos menores. Os três rodaram em um notebook com 32 GB. |
| **Disco** | 13, 17 ou 21 GB para o arquivo do modelo. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida do resto, como para o [Qwen3.6-35B-A3B](qwen36.pt-BR.md). Os arquivos de maternion não trazem
template de chat, então o instalador baixa o do próprio modelo na DiffuSpace. A licença do Clover é Apache 2.0.

## Qual modelo devo escolher?

Três tamanhos, todos feitos com o conversor do pull request 31306 do llama.cpp. O instalador escolhe Q3_K_M, ou
APEX-I-NANOPLUS num PC com menos RAM. Q4_K_M é escolhido pelo nome: `--model Q4_K_M`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **Q3_K_M** | o modelo inteiro com experts de 3 bits ([maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)) | 17 GB | cerca de 20 GB |
| **APEX-I-NANOPLUS** | experts de 2 e 3 bits, a partir de uma matriz de importância ([IsValorum](https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF)): o que pede menos RAM | 13 GB | cerca de 16 GB |
| **Q4_K_M** | experts de 4 bits (maternion) | 21 GB | cerca de 25 GB |

Os dois tamanhos de maternion foram quantizados sem matriz de importância. Quanto cada tamanho chega perto do modelo
treinado não foi medido.

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#clover-35b-a3b-preview).

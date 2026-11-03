# MiniCPM-V 4.7 35B-A3B

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**MiniCPM-V 4.7 35B-A3B** da OpenBMB, com a arquitetura do Qwen3.6-35B-A3B e pesos próprios, feito para ler imagens: um arquivo de 16 a 29 GB.

## Qual é a velocidade?

Medido como o [Qwen3.6-35B-A3B](qwen36.pt-BR.md), nos mesmos dois PCs. O arquivo dele não tem bloco de rascunho, então ele escreve um token por vez.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 64 tokens/s | 1,150 tokens/s |
| **mix_q3gud_im** | 59 tokens/s | 1,130-1,150 tokens/s |
| **Q6_K** | 40 tokens/s | 960-980 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q4_K_M** | 308-311 tokens/s | 7,910-7,950 tokens/s |
| **mix_q3gud_im** | 291-293 tokens/s | 7,730-7,740 tokens/s |
| **Q6_K** | 277-278 tokens/s | 7,290-7,330 tokens/s |

</td></tr>
</table>

Um prompt de 10,000 tokens é lido a 1,260 tokens/s no notebook e a 8,790 na estação de trabalho. No notebook o Q6_K
enche os 32 GB de RAM, e é isso que o deixa lento lá.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 3.5 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB e na placa de 8 GB de um notebook. Em uma placa AMD ele também roda, em beta, no Linux e no Windows: o instalador compila o motor no seu PC; ele rodou em uma RX 7600. |
| **RAM** | Cerca de **19, 25 ou 31 GB** ao lado de uma placa de 8 GB para os três tamanhos, pela estimativa do instalador. Uma placa de 32 GB guarda inteiros os três tamanhos. Os três tamanhos rodaram em um notebook com 32 GB. |
| **Disco** | 16, 21 ou 29 GB para o arquivo do modelo, 1 GB a mais para ler imagens. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. No Windows ele baixa o motor já compilado da
última versão publicada. Num processador sem AVX2, ou numa cópia cujo motor mudou depois dessa versão, ele compila o
motor no lugar, com o compilador de 2022 do Visual Studio e o CUDA Toolkit, e oferece instalar os dois. Ainda não:
duas placas no Windows, três placas ou mais. Em uma placa AMD no Windows o instalador compila o `strata-poly` para a
placa, como no Linux (as wheels TheRock da AMD em `.rocm-win`), e avisa que rodar ali é novo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Qual modelo devo escolher?

Três tamanhos por enquanto, da OpenBMB. O instalador escolhe Q4_K_M, ou mix_q3gud_im num PC com menos RAM. Q6_K é escolhido pelo nome: `--model Q6_K`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **Q4_K_M** | o modelo inteiro com experts de 4 bits ([OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)) | 21 GB | cerca de 25 GB |
| **mix_q3gud_im** | o mesmo com experts de 3 e 4 bits: o que pede menos RAM | 16 GB | cerca de 19 GB |
| **Q6_K** | o mesmo com experts de 6 bits | 29 GB | cerca de 31 GB |

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#minicpm-v-47-35b-a3b).

# Tiel Coder 35B-A3B

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Tiel Coder 35B-A3B**, o Ornith 1.5 quantizado por peculiar-ragdoll para agentes de programação: um arquivo de 14 a 23 GB. Ele também entende imagens.

## Qual é a velocidade?

Medido como o [Qwen3.6-35B-A3B](qwen36.pt-BR.md), nos mesmos dois PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 83 tokens/s | 1,700 tokens/s |
| **UD-IQ3_XXS** | 96-97 tokens/s | 1,610 tokens/s |
| **UD-Q4_K_XL** | 74-79 tokens/s | 1,660-1,670 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 337-345 tokens/s | 8,080-8,110 tokens/s |
| **UD-IQ3_XXS** | 352 tokens/s | 7,520 tokens/s |
| **UD-Q4_K_XL** | 316-322 tokens/s | 7,750-7,970 tokens/s |

</td></tr>
</table>

Um prompt de 10,000 tokens é lido a 1,670 tokens/s no notebook e a 6,300 na estação de trabalho.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 5 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB, nessa placa limitada a 8 GB e na placa de 8 GB de um notebook. Em uma placa AMD ele também roda, em beta, no Linux e no Windows: o instalador compila o motor no seu PC; usuários já o rodaram numa RX 6700 XT e numa RX 7800 XT no Linux, e numa RX 9070 XT no Windows. |
| **RAM** | Cerca de **18, 21 ou 27 GB** ao lado de uma placa de 8 GB para os três tamanhos, pela estimativa do instalador. Uma placa de 24 GB guarda inteiros os três tamanhos. Os três tamanhos rodaram em um notebook com 32 GB. |
| **Disco** | 14, 17 ou 23 GB para o arquivo do modelo, 1 GB a mais para ler imagens. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. No Windows ele baixa o motor já compilado da
última versão publicada. Num processador sem AVX2, ou numa cópia cujo motor mudou depois dessa versão, ele compila o
motor no lugar, com o compilador de 2022 do Visual Studio e o CUDA Toolkit, e oferece instalar os dois. Ainda não:
duas placas no Windows, três placas ou mais. Em uma placa AMD no Windows o instalador compila o `strata-poly` para a
placa, como no Linux (as wheels TheRock da AMD em `.rocm-win`), e avisa que rodar ali é novo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Qual modelo devo escolher?

Três tamanhos por enquanto. O instalador escolhe UD-Q3_K_XL, ou UD-IQ3_XXS num PC com menos RAM. UD-Q4_K_XL é escolhido pelo nome: `--model UD-Q4_K_XL`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q3_K_XL** | o modelo inteiro com experts de 3 e 4 bits ([peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)) | 17 GB | cerca de 21 GB |
| **UD-IQ3_XXS** | o mesmo com experts de 2 e 3 bits: o que pede menos RAM | 14 GB | cerca de 18 GB |
| **UD-Q4_K_XL** | o mesmo com experts de 4 a 6 bits | 23 GB | cerca de 27 GB |

Uma placa maior guarda mais do modelo e precisa de menos RAM: com UD-Q3_K_XL, cerca de 17 GB ao lado de uma placa de 12 GB, 13 GB ao lado de uma de 16 GB e 10 GB quando a placa guarda tudo. São estimativas do instalador. Elas não foram
conferidas num PC assim.

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#tiel-coder-35b-a3b).

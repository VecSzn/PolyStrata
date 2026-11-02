# Gemma 4 26B-A4B

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Gemma 4 26B-A4B**, um arquivo de 11 a 17 GB. Ele também entende imagens.

## Qual é a velocidade?

Medido como o [Qwen3.6-35B-A3B](qwen36.pt-BR.md), nos mesmos dois PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 92-93 tokens/s | 2,240 tokens/s |
| **UD-Q2_K_XL** | 101-104 tokens/s | 2,150-2,160 tokens/s |
| **UD-Q4_K_M** | 85 tokens/s | 2,150 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 353-356 tokens/s | 8,420-8,480 tokens/s |
| **UD-Q2_K_XL** | 332-338 tokens/s | 7,940-7,970 tokens/s |
| **UD-Q4_K_M** | 335-346 tokens/s | 8,320-8,390 tokens/s |
| **QAT-UD-Q4_K_XL** | 356-371 tokens/s | 8,460-8,680 tokens/s |

</td></tr>
</table>

Um prompt de 10,000 tokens é lido a 2,130 tokens/s no notebook e a 6,800 na estação de trabalho. QAT-UD-Q4_K_XL foi medido depois, com o motor da v0.1.45-poly.4; o UD-Q4_K_M, medido ao lado com esse motor, escreveu 333-353 tokens/s e leu 8,510-8,570. No notebook, com outros programas abertos e pouca RAM, ele rodou alternado com o UD-Q4_K_M: escreveu 50-62 tokens/s num chat contra 47-48, leu o prompt longo a 1,680-1,730 contra 1,660-1,670 e escreveu depois 65-68 contra 48-49.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 6 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB, nessa placa limitada a 8 GB e na placa de 8 GB de um notebook. Em uma placa AMD ele também roda, em beta, no Linux e no Windows: o instalador compila o motor no seu PC; usuários já o rodaram numa RX 6700 XT (12 GB) no Linux. |
| **RAM** | Cerca de **15, 18, 19 ou 21 GB** ao lado de uma placa de 8 GB para os quatro tamanhos, pela estimativa do instalador. Pela mesma estimativa, uma placa de 16 GB guarda inteiros os dois tamanhos menores e o arquivo QAT. Os quatro tamanhos rodaram em um notebook com 32 GB. |
| **Disco** | 11, 13, 15 ou 17 GB para o arquivo do modelo e o modelo de rascunho dele, 1 GB a mais para ler imagens. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. No Windows ele baixa o motor já compilado da
última versão publicada. Num processador sem AVX2, ou numa cópia cujo motor mudou depois dessa versão, ele compila o
motor no lugar, com o compilador de 2022 do Visual Studio e o CUDA Toolkit, e oferece instalar os dois. Ainda não:
duas placas no Windows, três placas ou mais. Em uma placa AMD no Windows o instalador compila o `strata-poly` para a
placa, como no Linux (as wheels TheRock da AMD em `.rocm-win`), e avisa que rodar ali é novo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Qual modelo devo escolher?

Quatro tamanhos por enquanto. O instalador escolhe UD-Q3_K_M, ou UD-Q2_K_XL num PC com menos RAM. Os outros dois são escolhidos pelo nome: `--model UD-Q4_K_M`, `--model QAT-UD-Q4_K_XL`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | o modelo inteiro com experts de 3 bits ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)), e o modelo de rascunho do Google para ele | 13 GB | cerca de 18 GB |
| **UD-Q2_K_XL** | o mesmo com experts de 2 bits: o que pede menos RAM | 11 GB | cerca de 15 GB |
| **UD-Q4_K_M** | o mesmo com experts de 4 e 5 bits | 17 GB | cerca de 21 GB |
| **QAT-UD-Q4_K_XL** | os pesos do Google treinados com quantização (QAT), em 4 bits ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)), com o próprio modelo de rascunho | 15 GB | cerca de 19 GB |

Uma placa maior guarda mais do modelo e precisa de menos RAM: com UD-Q3_K_M, cerca de 14 GB ao lado de uma placa de 12 GB e 10 GB quando a placa guarda tudo. São estimativas do instalador. Elas não foram
conferidas num PC assim.

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#gemma-4-26b-a4b).

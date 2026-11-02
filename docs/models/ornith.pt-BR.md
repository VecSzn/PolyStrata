# Ornith 1.5

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Ornith 1.5**, com a arquitetura do Qwen3.6-35B-A3B e outros pesos: um arquivo de 14 a 24 GB. Ele também entende imagens.

## Qual é a velocidade?

Medido como o [Qwen3.6-35B-A3B](qwen36.pt-BR.md), nos mesmos dois PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 76-82 tokens/s | 1,620 tokens/s |
| **APEX-I-MINI** | 80-86 tokens/s | 1,550-1,560 tokens/s |
| **APEX-I-QUALITY** | 59-63 tokens/s | 1,570-1,580 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 316-319 tokens/s | 8,040-8,110 tokens/s |
| **APEX-I-MINI** | 289-300 tokens/s | 7,710-7,740 tokens/s |
| **APEX-I-QUALITY** | 297-302 tokens/s | 7,860-7,880 tokens/s |

</td></tr>
</table>

Um prompt de 10,000 tokens é lido a 1,590 tokens/s no notebook e a 6,400 na estação de trabalho.

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 4 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB, nessa placa limitada a 8 GB e na placa de 8 GB de um notebook. Em uma placa AMD ele também roda, em beta, no Linux e no Windows: o instalador compila o motor no seu PC; testado até agora com Qwen3.6 e Ornith em uma RX 7600. |
| **RAM** | Cerca de **18, 22 ou 28 GB** ao lado de uma placa de 8 GB para os três tamanhos, pela estimativa do instalador. Uma placa de 24 GB guarda inteiros os dois tamanhos menores. Os três tamanhos rodaram em um notebook com 32 GB. |
| **Disco** | 14, 17 ou 24 GB para o arquivo do modelo, 1 GB a mais para ler imagens. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. No Windows ele baixa o motor já compilado da
última versão publicada. Num processador sem AVX2, ou numa cópia cujo motor mudou depois dessa versão, ele compila o
motor no lugar, com o compilador de 2022 do Visual Studio e o CUDA Toolkit, e oferece instalar os dois. Ainda não:
duas placas no Windows, três placas ou mais. Em uma placa AMD no Windows o instalador compila o `strata-poly` para a
placa, como no Linux (as wheels TheRock da AMD em `.rocm-win`), e avisa que rodar ali é novo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

## Qual modelo devo escolher?

Três tamanhos por enquanto. O instalador escolhe APEX-I-COMPACT, ou APEX-I-MINI num PC com menos RAM. APEX-I-QUALITY é escolhido pelo nome: `--model APEX-I-QUALITY`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | o modelo inteiro com experts de 3 e 4 bits ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF), de mudler) | 17 GB | cerca de 22 GB |
| **APEX-I-MINI** | o mesmo com experts de 2 e 3 bits: o que pede menos RAM | 14 GB | cerca de 18 GB |
| **APEX-I-QUALITY** | o mesmo com experts de 4 a 6 bits | 24 GB | cerca de 28 GB |

Uma placa maior guarda mais do modelo e precisa de menos RAM: com APEX-I-COMPACT, cerca de 17 GB ao lado de uma placa de 12 GB, 13 GB ao lado de uma de 16 GB e 10 GB quando a placa guarda tudo. São estimativas do instalador. Elas não foram
conferidas num PC assim.

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#ornith-15).

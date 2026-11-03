# Qwen3.6-35B-A3B

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Qwen3.6-35B-A3B**, um arquivo de 13 a 23 GB. Ele também entende imagens.

## Qual é a velocidade?

Medido pelo PolyStrata em dois PCs, com uma conversa curta e um prompt de 30,000 a 32,000 tokens: à esquerda um
notebook com uma placa de 8 GB, onde o processador calcula os experts que não cabem na placa, e à direita uma estação
de trabalho cuja placa guarda o modelo inteiro.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 94-100 tokens/s | 1,690 tokens/s |
| **UD-Q2_K_XL** | 123-124 tokens/s | 1,590 tokens/s |
| **UD-Q4_K_M** | 82-86 tokens/s | 1,560-1,570 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 394-395 tokens/s | 7,800-7,810 tokens/s |
| **UD-Q2_K_XL** | 450 tokens/s | 7,720-7,740 tokens/s |
| **UD-Q4_K_M** | 395-396 tokens/s | 8,070-8,080 tokens/s |

</td></tr>
</table>

Um prompt de 10,000 tokens é lido a 1,660 tokens/s no notebook e a 6,300 na estação de trabalho.
Os números da placa da estação limitada a 8
GB, um substituto de antes da medição no notebook, estão em [docs/BACKENDS.md](../BACKENDS.md#with-8-gb-of-vram) (em
inglês).

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 5 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB, nessa placa limitada a 8 GB e na placa de 8 GB de um notebook. Em uma placa AMD ele também roda, em beta, no Linux e no Windows: o instalador compila o motor no seu PC; testado até agora com Qwen3.6 e Ornith em uma RX 7600. |
| **RAM** | Cerca de **17, 21 ou 27 GB** ao lado de uma placa de 8 GB para os três tamanhos, pela estimativa do instalador. Uma placa de 24 GB guarda inteiros os dois tamanhos menores. Os três tamanhos rodaram em um notebook com 32 GB. |
| **Disco** | 13, 17 ou 23 GB para o arquivo do modelo, 1 GB a mais para ler imagens. |
| **Sistema** | Linux ou Windows 10 / 11, e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. No Windows ele baixa o motor já compilado da
última versão publicada. Num processador sem AVX2, ou numa cópia cujo motor mudou depois dessa versão, ele compila o
motor no lugar, com o compilador de 2022 do Visual Studio e o CUDA Toolkit, e oferece instalar os dois. Ainda não:
duas placas no Windows, três placas ou mais. Em uma placa AMD no Windows o instalador compila o `strata-poly` para a
placa, como no Linux (as wheels TheRock da AMD em `.rocm-win`), e avisa que rodar ali é novo
([AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)).

Desde a 0.1.44, o próprio motor do [Strata](https://github.com/Niko1221/Strata) também roda o Qwen3.6, a partir do
UD-Q4_K_XL (`--family qwen36-strata`, 32 GB de RAM ou mais). Numa RTX 5090 com UD-Q4_K_XL, ele leu um prompt de 4K
1,6 vez mais rápido que o strata-poly; num prompt de 31K e na escrita os dois ficaram quase iguais
([medido](../BACKENDS.md#qwen36-on-two-engines), em inglês). Os tamanhos menores, para placas de 8 GB, continuam no strata-poly.

## Qual modelo devo escolher?

Três tamanhos por enquanto. O instalador escolhe UD-Q3_K_M, ou UD-Q2_K_XL num PC com menos RAM. UD-Q4_K_M é escolhido pelo nome: `--model UD-Q4_K_M`.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | o modelo inteiro com experts de 3 bits ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 GB | cerca de 21 GB |
| **UD-Q2_K_XL** | o mesmo com experts de 2 e 3 bits: o que pede menos RAM | 13 GB | cerca de 17 GB |
| **UD-Q4_K_M** | o mesmo com experts de 4 bits | 23 GB | cerca de 27 GB |

Uma placa maior guarda mais do modelo e precisa de menos RAM: com UD-Q3_K_M, cerca de 17 GB ao lado de uma placa de 12 GB, 13 GB ao lado de uma de 16 GB e 10 GB quando a placa guarda tudo. São estimativas do instalador. Elas não foram
conferidas num PC assim.

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#qwen36-35b-a3b).

# GLM 5.3 Flash

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**GLM 5.3 Flash**, o maior: um arquivo de 79 a 114 GB, que fica na maior parte na RAM.

## Qual é a velocidade?

Medido para o PolyStrata numa estação de trabalho, com uma conversa curta e um prompt de 29K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **3.0BIT** | 59-66 tokens/s | 820 tokens/s |
| **UD-IQ2_XXS** | 50-52 tokens/s | 1,060-1,080 tokens/s |
| **REAP50-Q3_K_M** | 59-66 tokens/s | 1,300-1,310 tokens/s |

</td></tr>
</table>

O GLM 5.3 Flash é um modelo muito maior que os outros, e só cerca de um quinto dos experts dele cabe nos 32 GB desta placa, por isso
ele escreve mais devagar. O que foi medido e como: [docs/GLM.md](../GLM.md#what-it-was-measured-on) (em inglês).

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova. Ele só rodou com **32 GB de VRAM**. Cerca de 12 GB são ocupados antes do primeiro expert, então uma placa de 12 GB não consegue iniciá-lo. |
| **RAM** | Ao lado de uma placa de 32 GB, **96 GB ou mais** para o modelo inteiro, 82 GB para o tamanho de 2 bits, 60 GB para o tamanho com metade dos experts. Uma placa menor precisa de mais RAM ([quanto](#qual-modelo-devo-escolher)). Ele rodou com 120 GB. |
| **Disco** | 114 GB, 102 GB ou 79 GB para o arquivo do modelo, num SSD. |
| **Sistema** | Linux e um driver NVIDIA atual. |

O instalador cuida de todo o resto. Ele compila o motor no seu PC e oferece instalar o que falta para isso
(g++ e o CUDA Toolkit). Ainda não: placas AMD, duas placas no Windows, três placas ou mais, imagens.
Um usuário roda no Windows com uma placa de 24 GB e 128 GB de RAM ([relato](https://github.com/VecSzn/PolyStrata/issues/10)).

A lista completa: [docs/GLM.md](../GLM.md#what-you-need) (em inglês).

## Qual modelo devo escolher?

Três tamanhos por enquanto. O instalador escolhe o primeiro para o qual a sua RAM é suficiente.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 32 GB |
| --- | --- | ---: | ---: |
| **3.0BIT** | o modelo inteiro, a cerca de 3 bits ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 GB | 96 GB |
| **UD-IQ2_XXS** | o modelo inteiro, a cerca de 2 bits ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): lê um prompt mais rápido e escreve mais devagar | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | metade dos experts roteados removida ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF), de patrickbdevaney) | 79 GB | 60 GB |

A RAM necessária depende da sua placa de vídeo: os experts que não cabem na placa precisam ficar na RAM.

| Sua placa de vídeo | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | o PC em que ele rodou (120 GB de RAM) |
| **24 GB** | cerca de 104 GB | cerca de 90 GB | cerca de 68 GB | ainda não testado |
| **16 GB** | cerca de 113 GB | cerca de 99 GB | cerca de 77 GB | ainda não testado |
| **12 GB ou menos** | | | | não inicia |

Os números de RAM são a estimativa do instalador, a partir do único PC em que ele rodou. Com menos RAM ele inicia
mesmo assim, mas lê experts do SSD enquanto responde e fica muito mais lento. O instalador avisa e pergunta antes.

As outras quantizações do GLM 5.3 Flash ainda não são suportadas: [o que falta](../GLM.md#what-is-missing) (em inglês).

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#glm-53-flash).

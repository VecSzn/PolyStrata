# DeepSeek V4 Flash

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**DeepSeek V4 Flash**, um arquivo de 91 GB, que também fica na maior parte na RAM. A memória que ele guarda de uma conversa é pequena: um contexto de 128K ocupa só 0,7 GB de VRAM a mais que um de 32K.

## Qual é a velocidade?

Medido para o PolyStrata na mesma estação de trabalho que o [GLM 5.3 Flash](glm.pt-BR.md), com uma conversa curta e um prompt de 30K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-IQ2_XXS** | 56 tokens/s | 1,040 tokens/s |

</td></tr>
</table>

No mesmo PC, o llama.cpp escrevia 25 a 28 tokens por segundo e lia um prompt de 3,384 tokens a 96. Um prompt de
116,473 tokens, num contexto de 128K, foi lido e respondido em 174 segundos, e um de 487,529 tokens, num contexto de
512K, em 32 minutos; nas duas vezes os três fatos escondidos nele foram encontrados. O que foi medido e como: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-it-was-measured-on) (em inglês).

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova. Ele só rodou com **32 GB de VRAM**. Cerca de 11,5 GB são ocupados antes do primeiro expert, então uma placa de 12 GB não consegue iniciá-lo. |
| **RAM** | Ao lado de uma placa de 32 GB, **72 GB ou mais**, pela estimativa do instalador. Uma placa menor precisa de mais RAM ([quanto](#qual-modelo-devo-escolher)). Ele rodou com 120 GB. |
| **Disco** | 91 GB para os três arquivos do modelo, num SSD. |
| **Sistema** | Linux e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como no GLM 5.3 Flash. Ainda não testado: placas AMD, duas placas. Ele só
lê texto. Um usuário roda no Windows com uma placa de 24 GB e 128 GB de RAM ([relato](https://github.com/VecSzn/PolyStrata/issues/21)).

A lista completa: [docs/DEEPSEEK.md](../DEEPSEEK.md#what-you-need) (em inglês).

## Qual modelo devo escolher?

Um tamanho só, por enquanto.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 32 GB |
| --- | --- | ---: | ---: |
| **UD-IQ2_XXS** | o modelo inteiro, com cerca de 2 bits ([Unsloth](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)) | 91 GB | 72 GB |

Como no GLM 5.3 Flash, os experts que a placa não guarda ficam na RAM: pela estimativa do instalador, cerca de 80 GB ao
lado de uma placa de 24 GB e 89 GB ao lado de uma de 16 GB; uma placa de 12 GB não consegue iniciá-lo. Só a placa de
32 GB foi testada.

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#deepseek-v4-flash).

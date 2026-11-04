# Qwen3.8-Flash-Next

Voltar para [README.pt-BR.md](../../README.pt-BR.md)

**Qwen3.8-Flash-Next** numa placa de vídeo NVIDIA ou AMD de 12 GB ou mais, no Windows ou no Linux. Ele também entende imagens. O motor dele é o do [Strata](https://github.com/Niko1221/Strata), sem mudanças.

## Qual é a velocidade?

Medido pelo [Strata](https://github.com/Niko1221/Strata) em dois PCs gamer comuns, com um prompt de 32K tokens.

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB de RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 com o engine 0.1.36, as outras linhas com 0.1.26 (respostas de 4K, prompts de 32K). As tabelas
completas estão em [DETAILS.md](../DETAILS.md#speed-measured). Uma placa com mais VRAM é mais rápida: uma
RTX 3090 (24 GB) deve escrever cerca de 100-140 tokens por segundo. Conversas longas e outras placas:
[velocidade de cada modelo](../MODELS.md#how-fast-is-each-size), [resultados da comunidade](../COMMUNITY_BENCHMARKS.md).

## Do que você precisa

| | |
| --- | --- |
| **Placa de vídeo** | **NVIDIA** GeForce RTX séries 20, 30, 40 ou 50, ou **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 ou séries RX 6800 / 6900. Ela precisa de **12 GB de VRAM ou mais**. |
| **RAM** | 32 GB ou mais. Sua RAM decide [qual modelo](#qual-modelo-devo-escolher) cabe. Com 64 GB, roda qualquer tamanho. |
| **Disco** | Cerca de 80 GB livres. Use um SSD se puder: a primeira inicialização fica bem mais rápida. |
| **Sistema** | Windows 10 / 11 ou Linux, e um driver de vídeo atualizado da NVIDIA ou da AMD. |

O instalador cuida de todo o resto. Duas ou três placas podem dividir o modelo ([multi-GPU](../MULTI_GPU.md)).

Experimental, escrito e testado por membros da comunidade nas próprias máquinas:

- **Placas de vídeo mais antigas** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](../OLDER_GPUS.md).
- **Intel Arc**, compilado a partir do código-fonte no Linux: [Intel Arc](../INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, compilado a partir do código-fonte no Linux: [Strix Halo](../STRIX_HALO.md).
- **Processadores mais antigos sem AVX2**: funcionam, mas devagar. [Older CPUs](../INSTALL.md#older-cpus-experimental).

A lista completa: [docs/INSTALL.md](../INSTALL.md#what-you-need).

## Qual modelo devo escolher?

O instalador recomenda um de acordo com sua RAM. O mesmo modelo vem em vários tamanhos, mais ou menos comprimidos.
Os tamanhos menores são mais rápidos. Os maiores são um pouco mais inteligentes.

| Sua RAM | Escolha | Por quê |
| --- | --- | --- |
| **32 GB** | **Coder** | cabe em 32 GB e é feito para código (com uma placa de 24 GB, Q2_0 e IQ2_XS também rodam) |
| **48 GB** | **IQ2_XS** (ou Q2_0, o mais rápido) | os tamanhos maiores não cabem |
| **64 GB** | **IQ2_XS** (recomendado), ou IQ3_XXS / IQ3_S | todos os tamanhos cabem; IQ3_S é o melhor e o mais lento |
| **96 GB ou mais** | **IQ3_S**, ou o UD-IQ4_XS da Unsloth (~4-bit) | espaço para os maiores tamanhos com todo o resto aberto |

- **[Coder](../MODELS.md#coder):** uma versão para programação com metade dos experts removida. Ela alcança 91%
  da pontuação do modelo completo no SWE-bench Verified (medido pelos autores) e cabe em 32 GB de RAM. É mais fraca
  fora do código, inclusive em chinês e outros textos CJK (#438). Para isso, escolha Q2_0, IQ2_XS ou IQ3_S, que
  mantêm todos os experts.
- **[Swift 1.5](../MODELS.md#swift-15):** um fine-tune que pensa por bem menos tempo antes de responder. Você
  recebe a resposta mais cedo, com quase a mesma qualidade.
- **[Unsloth UD-IQ4_XS](../MODELS.md#unsloth-ud-iq4_xs):** a versão de ~4 bits da Unsloth, entre IQ3_S e
  UD-Q4_K_XL em qualidade. Um download de 94 GB. Com menos de ~80 GB de RAM, o PolyStrata lê uma parte dele do SSD
  enquanto responde, então ali ele fica mais lento (um SSD NVMe ajuda).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL e Q8_0](../MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):** de 4 a 8 bits, mais próximos do modelo
  completo (93-95% o mesmo próximo token que o Q8_0). Rápidos quando todos os especialistas cabem na RAM (~88 GB para
  o UD-Q4_K_XL até ~136 GB para o Q8_0); com menos RAM, o PolyStrata lê uma parte do SSD enquanto responde, o que é bem
  mais lento (7-8.5 tokens/s para o UD-Q4_K_XL num PC com 64 GB).
- **[OrcaRouter's Uncensored IQ3_XXS](../MODELS.md#orcarouter-uncensored-iq3_xxs):** você configura à mão. Ele
  não está no menu do instalador.

Tamanhos, downloads e o que cabe onde: [docs/MODELS.md](../MODELS.md). Para adicionar outro modelo depois, rode
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Como funciona?

O diagrama e como funciona: [HOW_IT_WORKS_MODELS.pt-BR.md](../HOW_IT_WORKS_MODELS.pt-BR.md#qwen38-flash-next).

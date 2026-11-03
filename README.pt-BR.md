<h1 align="center">PolyStrata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

<p align="center"><b>Rode modelos de IA grandes no seu próprio PC</b><br>
Qwen3.8-Flash-Next, GLM 5.3 Flash, DeepSeek V4 Flash, LFM2.5, Flint, Qwen3.6, Clover, Ornith 1.5, Gemma 4, Tiel Coder, MiniCPM-V e Kolibri 1 · gratuito e de código aberto</p>

<p align="center"><img src="docs/media/speed.svg" width="820" alt="Tokens por segundo ao escrever uma resposta: 94 a 100 com o Qwen3.6-35B-A3B em um notebook com placa de 8 GB, 394 a 395 em uma RTX 5090"></p>

<p align="center"><img src="docs/media/claude-code-demo.gif" width="820" alt="Claude Code em um terminal ao lado do API Monitor do PolyStrata"><br>
<sub>Claude Code com Qwen3.6-35B-A3B UD-Q3_K_M no notebook com a placa de 8 GB, em tempo real: ele adiciona uma opção <code>--json</code> a um programa pequeno, escreve um teste e roda os testes. O servidor tinha lido o prompt de sistema do Claude Code ao iniciar (<code>"warm_start": true</code>).</sub></p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Um jardim com pagode em voxels que o modelo do PolyStrata escreveu, rodando no navegador"></a><br>
<sub>Um jardim com pagode em voxels, feito com um único prompt numa RTX 5070 com o <a href="https://github.com/Niko1221/Strata">Strata</a> (Qwen3.8-Flash-Next IQ3_S, contexto de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vídeo completo (49 s)</a></sub></p>

O PolyStrata roda modelos de IA grandes num PC comum, modelos que normalmente precisam de um servidor.
Eles conversam, escrevem código e funcionam com seus apps e agentes de programação. Nada sai do seu PC.

Ele roda doze modelos, com dois motores:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** numa placa de vídeo NVIDIA ou AMD de
  12 GB ou mais, no Windows ou no Linux. Ele também entende imagens. O motor dele é o do
  [Strata](https://github.com/Niko1221/Strata), sem mudanças.

Os outros onze usam um motor escrito para o PolyStrata, com uma placa NVIDIA ou [duas](docs/MULTI_GPU.md#the-strata-poly-models): os onze no Linux, e todos menos o GLM 5.3 Flash e o
DeepSeek V4 Flash também no Windows. São experimentais: só rodaram em um PC com uma placa de 32 GB, e os oito menores também em um
notebook com uma placa de 8 GB.

- **[GLM 5.3 Flash](docs/models/glm.pt-BR.md)**, o maior: um arquivo de 79 a 114 GB, que fica na maior parte na RAM.
- **[DeepSeek V4 Flash](docs/models/deepseek.pt-BR.md)**, um arquivo de 91 GB, que também fica na maior parte na RAM. A memória que
  ele guarda de uma conversa é pequena: um contexto de 128K ocupa só 0,7 GB de VRAM a mais que um de 32K.
- **[LFM2.5-24B-A2B](docs/models/lfm25.pt-BR.md)** da Liquid AI, um arquivo de 11 a 25 GB. A maioria das camadas dele não tem
  atenção, então uma conversa longa ocupa pouca VRAM: um contexto de 128K ocupa só 2 GB a mais que 32K. A licença
  dele só permite o uso comercial a empresas que faturam menos de 10 milhões de dólares por ano.
- **[Flint 40B-A2.5B](docs/models/flint.pt-BR.md)** da Recursive, feito para agentes de código: um arquivo de 18 a 42 GB. A
  maioria das camadas dele só olha 513 tokens para trás, então uma conversa longa ocupa pouca VRAM: um contexto de
  128K ocupa só 2,6 GB a mais que 32K.
- **[Qwen3.6-35B-A3B](docs/models/qwen36.pt-BR.md)**, um arquivo de 13 a 23 GB. Ele também entende imagens.
- **[Clover-35B-A3B-Preview](docs/models/clover.pt-BR.md)** da DiffuSpace, o Qwen3.6-35B-A3B treinado mais para
  escrever também um bloco de tokens de uma vez: um arquivo de 13 a 21 GB. Ele só lê texto.
- **[Ornith 1.5](docs/models/ornith.pt-BR.md)**, com a arquitetura do Qwen3.6-35B-A3B e outros pesos: um arquivo de 14 a 24 GB. Ele
  também entende imagens.
- **[Gemma 4 26B-A4B](docs/models/gemma4.pt-BR.md)**, um arquivo de 11 a 17 GB. Ele também entende imagens.
- **[Tiel Coder 35B-A3B](docs/models/tiel.pt-BR.md)**, o Ornith 1.5 quantizado por peculiar-ragdoll para agentes de
  programação: um arquivo de 14 a 23 GB. Ele também entende imagens.
- **[MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.pt-BR.md)** da OpenBMB, com a arquitetura do Qwen3.6-35B-A3B e pesos próprios,
  feito para ler imagens: um arquivo de 16 a 29 GB.
- **[Kolibri 1](docs/models/kolibri.pt-BR.md)** da Aleph Alpha, que escreve alemão tão bem quanto inglês: um arquivo de
  29 a 48 GB, que fica quase todo na RAM ao lado de uma placa pequena.

O servidor pode trocar entre os modelos instalados enquanto está rodando
([como](docs/BACKENDS.md#changing-the-model), em inglês).

O PolyStrata é feito sobre o [Strata](https://github.com/Niko1221/Strata) de Niko1221.
Por que é um repositório separado, placas AMD e menores, fine-tunes e outras perguntas: [FAQ](docs/FAQ.md) (em inglês).

## Qual é a velocidade?

Um token é mais ou menos ¾ de uma palavra.

- **Escreve respostas:** a velocidade com que a resposta aparece numa conversa curta. 60 tokens por segundo é mais rápido do que você consegue ler.
- **Lê seu prompt:** a velocidade com que ele absorve o que você envia (aqui, um documento, código ou histórico de conversa de 32K tokens).

A tabela mostra o tamanho que o instalador escolhe para cada modelo, em tokens por segundo. Os números de cada tamanho, a comparação com o llama.cpp e como foi medido estão na página de cada modelo.

| Modelo | Notebook com placa de 8 GB<br>escreve / lê | RTX 5090<br>escreve / lê | Todos os tamanhos |
| --- | ---: | ---: | --- |
| LFM2.5-24B-A2B | 70-73 / 3,590-3,620 | 480-485 / 23,390-23,460 | [LFM2.5](docs/models/lfm25.pt-BR.md) |
| Flint 40B-A2.5B | 61-63 / 2,000-2,010 | 251-252 / 12,900 | [Flint](docs/models/flint.pt-BR.md) |
| Qwen3.6-35B-A3B | 94-100 / 1,690 | 394-395 / 7,800-7,810 | [Qwen3.6](docs/models/qwen36.pt-BR.md) |
| Clover-35B-A3B-Preview | 68-73 / 1,560-1,700 | não medido | [Clover](docs/models/clover.pt-BR.md) |
| Ornith 1.5 | 76-82 / 1,620 | 316-319 / 8,040-8,110 | [Ornith](docs/models/ornith.pt-BR.md) |
| Gemma 4 26B-A4B | 92-93 / 2,240 | 353-356 / 8,420-8,480 | [Gemma 4](docs/models/gemma4.pt-BR.md) |
| Tiel Coder 35B-A3B | 83 / 1,700 | 337-345 / 8,080-8,110 | [Tiel Coder](docs/models/tiel.pt-BR.md) |
| MiniCPM-V 4.7 35B-A3B | 64 / 1,150 | 308-311 / 7,910-7,950 | [MiniCPM-V](docs/models/minicpmv.pt-BR.md) |
| Kolibri 1 | não medido | 189-192 / 4,980-5,070 | [Kolibri 1](docs/models/kolibri.pt-BR.md) |
| GLM 5.3 Flash | não roda: precisa de uma placa de 32 GB | 59-66 / 820 | [GLM](docs/models/glm.pt-BR.md) |
| DeepSeek V4 Flash | não roda: precisa de uma placa de 32 GB | 56 / 1,040 | [DeepSeek](docs/models/deepseek.pt-BR.md) |

Notebook: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB de RAM. Estação de trabalho: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 90 a 120 GB de RAM. O Flint é UD-Q3_K_XL no notebook, o tamanho que o instalador escolhe com 32 GB de RAM, e UD-Q4_K_XL na estação de trabalho. O Kolibri 1 é Q2_K na estação de trabalho; o instalador escolhe ali o Q4_K_M, que não foi medido.

O Qwen3.8-Flash-Next roda no motor do projeto [Strata](https://github.com/Niko1221/Strata) e foi medido por esse projeto num PC com uma RTX 5070 (12 GB) e 64 GB de RAM: IQ2_XS escreve a 79 e lê um prompt de 32K a 2,090 ([todos os tamanhos](docs/models/qwen38.pt-BR.md)).

Os onze modelos do motor próprio do PolyStrata foram medidos com o benchmark dele, que você pode rodar no seu PC:
`python polystrata.py benchmark --model <name> --prompt-tokens 24000`
([o que ele envia e mostra](docs/BACKENDS.md#benchmark), em inglês).
O que outras pessoas mediram nas próprias placas, e como comparar com o llama.cpp na sua:
[resultados da comunidade](docs/COMMUNITY_BENCHMARKS.md#the-models-polystrata-adds) (em inglês).
No notebook, o PolyStrata leu os prompts de 2.7 a 3.5 vezes mais rápido que o llama.cpp b11540 e escreveu de
1.5 a 1.8 vezes mais rápido, com o Qwen3.6 e com o Tiel Coder ([as duas medições](docs/COMMUNITY_BENCHMARKS.md#comparing-with-llamacpp), em inglês).

## Quanta RAM ele ocupa

Desde a 0.1.45-poly.7, o motor dos onze modelos ocupa menos RAM: ele não guarda mais na memória as partes do arquivo
do modelo que já estão na placa de vídeo. No notebook, depois de um prompt de 32,000 tokens, o Qwen3.6-35B-A3B ocupava
17.9 GiB com a 0.1.45-poly.6 e agora ocupa 13.1 GiB, escrevendo na mesma velocidade. No Linux, foi de 17.8 para
12.6 GiB.

Para ocupar menos, adicione `"--ram-experts-mib", "6144"` à lista `"args"` do `strata-<model>.json`
([onde](docs/INSTALL.md#where-things-are-stored)) e reinicie. O motor passa a guardar essa quantidade de MiB de RAM
para os experts que ficam fora da VRAM, também enquanto responde, e avisa ao iniciar. Os outros ficam no cache de
arquivos do sistema, onde seus outros programas podem usar essa memória, e são lidos de lá quando um token precisa
deles. Vem desligado porque escreve mais devagar, e seguimos trabalhando nisso. No notebook (RTX 4070 Laptop 8 GB,
32 GB de RAM, Windows 11), Qwen3.6-35B-A3B UD-Q3_K_M:

| | padrão | `--ram-experts-mib 6144` |
| --- | ---: | ---: |
| RAM que o motor ocupa depois de carregar | 11.7 GiB | **6.5 GiB** |
| no máximo enquanto escreve: chat curto / depois de um prompt de 32K | 14.3 / 14.0 GiB | **6.7 / 7.3 GiB** |
| entre pedidos | 12.8 GiB | **7.1 GiB** |
| Escreve, tokens/s: o primeiro chat curto | 71.6 | 69.4 (3% mais devagar) |
| depois do prompt de 32K | 65.4 | 63.7 (3% mais devagar) |
| um chat curto depois disso | 84.2 | 72.9 (13% mais devagar) |
| uma resposta longa (1,024 tokens de um chat curto) | 90.6 | 78.3 (14% mais devagar) |

Nesta medição outros programas ocupavam parte da VRAM, então as duas colunas escrevem mais devagar do que com a placa
livre. Em outras duas medições escreveu de 6 a 8% mais devagar depois do prompt de 32K e de 8 a 10% no chat seguinte.
No Linux (uma RTX PRO 6000 limitada a cerca de 8 GB) ocupou de 7.0 a 7.7 GiB em vez de 12.3 a 14.3 e escreveu de 7 a
14% mais devagar. Com o cache de experts fixo, como uma versão é conferida, as respostas são iguais token a token.

O motor também precisa de 350 a 400 MiB a menos de VRAM para ler prompts: o cache de experts fica com eles (o Qwen3.6
mantém 2,712 experts na VRAM em vez de 2,444), ou `--vram-reserve-mib` os deixa para outros programas.
Como foi medido: [docs/BACKENDS.md](docs/BACKENDS.md#less-of-the-pcs-ram-and-vram).

## Do que você precisa

| Modelo | Placa de vídeo | RAM (tamanho do instalador) | Disco | Sistema |
| --- | --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/models/qwen38.pt-BR.md) | NVIDIA RTX séries 20 a 50, ou AMD RX 6800 / 6900, RX 7700 XT / 7800 XT / 7900 XT / XTX, RX 9060 XT / 9070 / 9070 XT, Radeon AI PRO R9700; 12 GB ou mais | 32 GB ou mais; com 64 GB, roda qualquer tamanho | cerca de 80 GB | Windows, Linux |
| [GLM 5.3 Flash](docs/models/glm.pt-BR.md) | NVIDIA RTX série 20 ou mais nova; só testado numa placa de 32 GB | 96 GB ao lado de uma placa de 32 GB | 79 a 114 GB, num SSD | Linux |
| [DeepSeek V4 Flash](docs/models/deepseek.pt-BR.md) | igual ao de cima | 72 GB ao lado de uma placa de 32 GB | 91 GB, num SSD | Linux |
| [LFM2.5-24B-A2B](docs/models/lfm25.pt-BR.md) | NVIDIA RTX série 20 ou mais nova, 8 GB ou mais | cerca de 20 GB ao lado de uma placa de 8 GB | 11 a 25 GB | Windows, Linux |
| [Flint 40B-A2.5B](docs/models/flint.pt-BR.md) | igual ao de cima | cerca de 22 GB ao lado de uma placa de 8 GB | 18 a 42 GB | Windows, Linux |
| [Qwen3.6-35B-A3B](docs/models/qwen36.pt-BR.md) | igual ao de cima; AMD em beta | cerca de 21 GB ao lado de uma placa de 8 GB | 13 a 23 GB | Windows, Linux |
| [Clover-35B-A3B-Preview](docs/models/clover.pt-BR.md) | NVIDIA RTX série 20 ou mais nova, 8 GB ou mais | cerca de 20 GB ao lado de uma placa de 8 GB | 13 a 21 GB | Windows, Linux |
| [Ornith 1.5](docs/models/ornith.pt-BR.md) | igual ao de cima; AMD em beta | cerca de 22 GB ao lado de uma placa de 8 GB | 14 a 24 GB | Windows, Linux |
| [Gemma 4 26B-A4B](docs/models/gemma4.pt-BR.md) | igual ao de cima; AMD em beta | cerca de 18 GB ao lado de uma placa de 8 GB | 11 a 17 GB | Windows, Linux |
| [Tiel Coder 35B-A3B](docs/models/tiel.pt-BR.md) | igual ao de cima; AMD em beta | cerca de 21 GB ao lado de uma placa de 8 GB | 14 a 23 GB | Windows, Linux |
| [MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.pt-BR.md) | igual ao de cima; AMD em beta | cerca de 25 GB ao lado de uma placa de 8 GB | 16 a 29 GB | Windows, Linux |
| [Kolibri 1](docs/models/kolibri.pt-BR.md) | NVIDIA RTX série 20 ou mais nova, 8 GB ou mais | cerca de 33 GB ao lado de uma placa de 8 GB | 29 a 48 GB | Windows, Linux |

A RAM é a estimativa do instalador; uma placa maior precisa de menos. O instalador cuida de todo o resto: no Windows ele baixa o motor já compilado e só compila no seu PC num processador sem AVX2 ou quando o motor da sua cópia é mais novo que a última versão, e oferece instalar as ferramentas que faltam. Duas ou três placas podem dividir um modelo ([multi-GPU](docs/MULTI_GPU.md)). Placas de vídeo mais antigas, Intel Arc, Strix Halo e processadores antigos sem AVX2: [a lista completa](docs/INSTALL.md#what-you-need). A placa, a RAM e os tamanhos de cada modelo em detalhe estão na página dele.

Do que cada um dos seis (Qwen3.6, Clover, Ornith 1.5, Gemma 4, Tiel Coder e MiniCPM-V) precisa e como foi conferido: [docs/BACKENDS.md](docs/BACKENDS.md#install) (em inglês).

## Instalação

### Deixe sua IA instalar

Você usa um assistente de programação com IA (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Cole isto nele:

```text
Set up PolyStrata on this PC for me: https://github.com/VecSzn/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

Ele verifica sua placa de vídeo, RAM e disco e escolhe o modelo que cabe. Depois instala, inicia e explica como
conectar seus apps. Ferramentas de IA também podem instalar, iniciar e parar o PolyStrata pelo
[servidor MCP](docs/MCP_SERVER.md) dele.

### Ou faça você mesmo

[Baixe o PolyStrata](https://github.com/VecSzn/PolyStrata/archive/refs/heads/main.zip) e descompacte (ou use `git clone`).
**Windows:** clique duas vezes em **`START-HERE.bat`**. **Linux:** rode **`./setup.sh`** na pasta do PolyStrata.

**Mac (Apple Silicon, M1 a M5, macOS 15 ou mais recente):** instale uma vez as Command Line Tools da Apple (`xcode-select --install` no Terminal; o Xcode em si não é necessário). Depois abra o Terminal na pasta do PolyStrata e rode **`./setup.sh`**. Ele compila o motor do Mac no seu Mac (uns 7 minutos), faz as mesmas perguntas, baixa o modelo e o inicia. Da próxima vez, `./setup.sh` inicia na hora. Qual tamanho cabe no seu Mac e qual a velocidade: [docs/MAC.md](docs/MAC.md). O motor do Mac roda só o Qwen3.8-Flash-Next; os outros onze modelos precisam de uma placa NVIDIA ou AMD.

Os passos são os mesmos para NVIDIA e AMD. O instalador encontra sua placa e configura o engine certo para ela.
Ele faz algumas perguntas:

- qual modelo e qual tamanho,
- quanto contexto (quanto texto o modelo guarda na memória),
- se ele deve entender imagens (o GLM 5.3 Flash, o DeepSeek V4 Flash, o LFM2.5, o Flint, o Clover e o Kolibri 1 não entendem).

Aperte Enter em cada uma para usar a resposta recomendada. Depois ele baixa o modelo (cerca de 70 GB para o
Qwen3.8-Flash-Next, 79 a 114 GB para o GLM 5.3 Flash, 91 GB para o DeepSeek V4 Flash, 29 a 48 GB para o Kolibri 1, 11 a 42 GB para os outros oito) e o inicia.
Se o download parar, rode de novo: ele continua de onde parou. Seu navegador abre o app do PolyStrata em
`http://127.0.0.1:8080`.

Os doze estão na lista de modelos do instalador. `./setup.sh --family glm`, `deepseek`, `lfm25`, `flint`, `qwen36`, `clover`, `ornith`,
`gemma4`, `tiel`, `minicpmv` ou `kolibri` vai direto para um deles ([o que ele pergunta e grava](docs/GLM.md#install), em inglês).

> **Enquanto o modelo inicia, seu PC pode ficar lento ou parar de responder por 1-3 minutos** (mais na primeira vez).
> O PolyStrata carrega 35-55 GB na sua RAM e reserva parte dela para a placa de vídeo. Isso é normal. Espere e não
> feche a janela. A janela mostra o que o PolyStrata está fazendo.

**Da próxima vez**, rode `START-HERE.bat` (ou `./setup.sh`) de novo. Ele inicia na hora e não baixa nada duas
vezes. Feche a janela para parar o modelo. `UPDATE.bat` (`./update.sh`) atualiza o PolyStrata sem iniciá-lo.
Atualização, Docker, várias placas, onde ficam os arquivos e todas as opções: [docs/INSTALL.md](docs/INSTALL.md).

**É rápido no seu PC?** Rode `BENCH.bat` (`./bench.sh`): uns 5-10 minutos; ele mede o chat, prompts de 4K/16K e uma sessão no estilo agente com as suas próprias configurações e escreve um relatório para compartilhar ([como](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Qual modelo devo escolher?

O mesmo modelo vem em vários tamanhos, mais ou menos comprimidos: os menores são mais rápidos e precisam de menos RAM, os maiores ficam mais perto do original. O instalador escolhe um para a sua RAM e a sua placa quando você aperta Enter; para outro tamanho, adicione `--model <tamanho>` ao instalar.

| Modelo | O instalador escolhe | Com menos RAM | Só pelo nome |
| --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/models/qwen38.pt-BR.md) | 32 GB de RAM: Coder; 48 a 64 GB: IQ2_XS; 96 GB ou mais: IQ3_S |  | Q2_0, IQ3_XXS, os arquivos de 4 a 8 bits da Unsloth |
| [GLM 5.3 Flash](docs/models/glm.pt-BR.md) | 3.0BIT | UD-IQ2_XXS, depois REAP50-Q3_K_M |  |
| [DeepSeek V4 Flash](docs/models/deepseek.pt-BR.md) | UD-IQ2_XXS (o único tamanho) |  |  |
| [LFM2.5-24B-A2B](docs/models/lfm25.pt-BR.md) | UD-Q4_K_XL | UD-Q3_K_XL | QAD-Q4_0, UD-Q6_K_XL, Q8_0 |
| [Flint 40B-A2.5B](docs/models/flint.pt-BR.md) | UD-Q4_K_XL a partir de cerca de 40 GB de RAM | UD-Q3_K_XL | UD-Q6_K_XL, Q8_0 |
| [Qwen3.6-35B-A3B](docs/models/qwen36.pt-BR.md) | UD-Q3_K_M | UD-Q2_K_XL | UD-Q4_K_M |
| [Clover-35B-A3B-Preview](docs/models/clover.pt-BR.md) | Q3_K_M | APEX-I-NANOPLUS | Q4_K_M |
| [Ornith 1.5](docs/models/ornith.pt-BR.md) | APEX-I-COMPACT | APEX-I-MINI | APEX-I-QUALITY |
| [Gemma 4 26B-A4B](docs/models/gemma4.pt-BR.md) | UD-Q3_K_M | UD-Q2_K_XL | UD-Q4_K_M, QAT-UD-Q4_K_XL |
| [Tiel Coder 35B-A3B](docs/models/tiel.pt-BR.md) | UD-Q3_K_XL | UD-IQ3_XXS | UD-Q4_K_XL |
| [MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.pt-BR.md) | Q4_K_M | mix_q3gud_im | Q6_K |
| [Kolibri 1](docs/models/kolibri.pt-BR.md) | Q4_K_M a partir de cerca de 48 GB de RAM | Q2_K | Q3_K_M |

O que é cada tamanho, o download, a RAM que ele precisa ao lado de outras placas e quão perto chega do maior: a página de cada modelo, com link na tabela de “Qual é a velocidade?”.

## Como usar

Os doze modelos são usados do mesmo jeito: o mesmo app, a mesma API.

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="A aba Monitor do app PolyStrata ao lado de um agente de programação"><br>
<sub>O <b>Monitor</b> do app PolyStrata (à esquerda) enquanto um agente de programação escreve o jardim com pagode do vídeo (à direita). Uma captura de tela do <a href="https://github.com/Niko1221/Strata">Strata</a>.</sub></p>

- **No navegador:** abra `http://127.0.0.1:8080`. Lá tem o **Chat**, um **Monitor** ao vivo do modelo e da sua
  GPU/CPU/RAM, e o **About** com as configurações e os endereços.
- **Seus apps e agentes de programação:** adicione um provedor "OpenAI-compatible" com a URL base
  **`http://127.0.0.1:8080/v1`**. Qualquer chave de API e qualquer nome de modelo funcionam.
  - Apps que usam a API da Anthropic: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, e `"fit_max_tokens": true` em `strata-<model>.json`
    ([por quê](docs/FAQ.md#can-i-use-it-with-claude-code-codex-or-other-apps))).
  - Codex CLI e outros apps que usam a OpenAI Responses API: `/v1/responses`
    ([configuração](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Raciocínio:** escolha **off, low, medium ou high** no menu do chat ou no "reasoning effort" do seu app. Off é o
  mais rápido. High é o melhor para perguntas difíceis.
- **Do seu celular ou de outro PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Sempre defina uma chave.

Onde eles diferem:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash | DeepSeek V4 Flash | LFM2.5-24B-A2B | Flint 40B-A2.5B | Kolibri 1 | Qwen3.6-35B-A3B, Clover-35B-A3B-Preview, Ornith 1.5, Gemma 4 26B-A4B, Tiel Coder 35B-A3B, MiniCPM-V 4.7 35B-A3B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Imagens** | Responda sim para "Images?" na instalação. Depois clique em **Picture** no chat, ou anexe imagens no seu app. Placas AMD entendem imagens no Linux usando o processador; no Windows, ainda não. | Ainda não. | Ele só lê texto. | Ele só lê texto. | Ele só lê texto. | Ele só lê texto. | Responda sim para "Images?" na instalação, e depois como no Qwen3.8-Flash-Next. Numa placa com menos de 16 GB, o codificador de imagens roda no processador, e uma imagem leva alguns segundos. O Clover só lê texto. |
| **Uma requisição por vez** | Por padrão, o PolyStrata responde uma requisição e as outras esperam. Para responder várias ao mesmo tempo, defina `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Numa placa de 12 GB, isso deixa cada resposta mais lenta. | Uma por vez, as outras esperam. | Uma por vez, as outras esperam. | Uma por vez, as outras esperam. | Uma por vez, as outras esperam. | Uma por vez, as outras esperam. | Uma por vez, as outras esperam. |
| **Várias conversas** | A última é mantida, então as mensagens seguintes começam em segundos. Dois argumentos do motor mantêm várias na RAM ([detalhes](docs/DETAILS.md#using-it)). | A última é mantida. A instalação pergunta se deve manter várias na RAM, para que um agente e seus subagentes se revezem sem que os prompts sejam lidos de novo ([como](docs/GLM.md#several-conversations-at-once)). | Como o GLM 5.3 Flash. | Como o GLM 5.3 Flash. | Como o GLM 5.3 Flash. | Como o GLM 5.3 Flash. | Como o GLM 5.3 Flash. |
| **Prompts longos** | O PolyStrata lê a primeira mensagem de uma conversa inteira, cerca de 1 minuto a cada 30,000 tokens. As mensagens seguintes começam em segundos. | A primeira mensagem é lida inteira, cerca de 40 segundos a cada 30,000 tokens. | A primeira mensagem é lida inteira, cerca de 29 segundos a cada 30,000 tokens. | A primeira mensagem é lida inteira: 30,000 tokens levam 1,3 a 1,5 segundo na estação de trabalho e 8 a 10 segundos no notebook de 8 GB. | A primeira mensagem é lida inteira: 30,000 tokens levam cerca de 2,3 segundos na estação de trabalho e 15 a 22 segundos no notebook de 8 GB. | A primeira mensagem é lida inteira: 30,000 tokens levam cerca de 6 segundos na estação de trabalho com Q2_K. | A primeira mensagem é lida por inteiro: 30,000 tokens levam 4 a 5 segundos na estação de trabalho e 14 a 27 segundos no notebook de 8 GB. |

Mais: [onde suas conversas ficam salvas](docs/INSTALL.md#where-things-are-stored), [a API](docs/DETAILS.md#using-it).

### Opcional: mais velocidade com uma placa quando o modelo inteiro cabe na RAM

Duas configurações opcionais (de RubenTorney, #1904 e #1905). Elas ajudam quando você usa uma placa de vídeo e sua RAM guarda todos os experts (o modo arena do PolyStrata, por exemplo Q2_0 ou IQ3_XXS num PC com 64 GB). Abra sua configuração `strata-<model>.json` na pasta de dados do PolyStrata ([onde](docs/INSTALL.md#where-things-are-stored)) e adicione:

- **Geração mais rápida:** `"--adapt-async", "1"` na lista `"args"`. O cache de experts passa a ser atualizado numa thread auxiliar em vez de pausar a resposta. Medido numa RTX 5070 (Windows, Q2_0): +4-5% de geração, +7-10% em código e documentos longos. As respostas podem diferir um pouco do padrão (ficam iguais de uma execução para outra).
- **Leitura do prompt mais rápida no Windows:** `"STRATA_PIN_BY_NEED": "1"` no bloco `"env"`. O Windows só consegue travar cerca de metade da RAM para a placa; isto trava primeiro os experts de que a placa realmente precisa. Medido na RTX 5070: +4,4% na leitura do prompt.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Reinicie o PolyStrata (`START-HERE.bat` ou `./setup.sh`) depois da mudança. Para desfazer, remova as duas entradas.

## Algo deu errado?

- **Meu PC travou na primeira vez que o PolyStrata iniciou.** Isso é normal enquanto ele carrega o modelo. Espere e não
  feche a janela. Ainda travado depois de 10 minutos? Reinicie o PC, feche outros programas e tente de novo, ou
  escolha um tamanho menor.
- **Parou durante o download ou a instalação.** Rode `START-HERE.bat` (ou `./setup.sh`) de novo. Ele continua de
  onde parou.
- **Está muito lento e a luz do disco não para de piscar, ou aparece "the engine stopped unexpectedly".** Seu PC
  não tem RAM livre suficiente. Feche outros programas (navegadores usam muita), ou escolha um tamanho menor (Q2_0
  ou IQ2_XS).
- **Diz que a porta 8080 já está em uso.** O PolyStrata já está rodando. Procure a janela dele.

Mais problemas e como resolver: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Ainda sem solução? Abra uma
[issue](https://github.com/VecSzn/PolyStrata/issues) e anexe o `strata-<model>.log` da pasta do PolyStrata. Encontrou um
problema de segurança? Informe em particular: [SECURITY.md](SECURITY.md).

## Como funciona?

Modelos como este normalmente rodam em servidores com centenas de gigabytes de memória de vídeo. Sua placa de vídeo
tem 8-32 GB. O PolyStrata faz o modelo caber **dividindo o trabalho pelo PC inteiro**. Pense numa cozinha: o que você
usa o tempo todo fica na bancada, e o resto espera na despensa.

O diagrama e a explicação de cada modelo: [docs/HOW_IT_WORKS_MODELS.pt-BR.md](docs/HOW_IT_WORKS_MODELS.pt-BR.md).

## Créditos e licença

O Qwen3.8-Flash-Next é da [equipe Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Ele foi comprimido por
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) e Unsloth.
O GLM 5.3 Flash roda a partir dos arquivos de [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF),
da Unsloth e de patrickbdevaney. O Qwen3.6-35B-A3B é da equipe Qwen, o Gemma 4 do Google e o DeepSeek V4 Flash da DeepSeek, os três na quantização da
[Unsloth](https://huggingface.co/unsloth). O LFM2.5-24B-A2B é da
[Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), na quantização da Unsloth e no QAD-Q4_0 da própria
Liquid AI. O Flint 40B-A2.5B é da [Recursive](https://huggingface.co/RecursiveSI/flint), na quantização
da Unsloth. O Kolibri 1 é da [Aleph Alpha](https://huggingface.co/Aleph-Alpha/Kolibri-1), na conversão GGUF da [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF). O Clover-35B-A3B-Preview é da [DiffuSpace](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview), na quantização de [maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF) e no APEX de IsValorum. O Ornith 1.5 é da ornith-ai, na quantização APEX de
[mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF). O Tiel Coder 35B-A3B é o Ornith 1.5 na quantização de
[peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP);
[u/Sensitive_Song4219](https://www.reddit.com/r/LocalLLM/comments/1x22inc/comment/pf3r111/) descobriu que ele roda no PolyStrata. O MiniCPM-V 4.7 35B-A3B é da [OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf),
na quantização dela mesma; [iMilnb](https://github.com/iMilnb) o trouxe para o PolyStrata. O PolyStrata é feito sobre o [Strata](https://github.com/Niko1221/Strata) de Niko1221 e seus colaboradores,
e usa partes do [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Todos os créditos:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). O PolyStrata é de código aberto sob a [MIT License](LICENSE).
Algumas partes e todos os modelos têm licenças próprias ([quais](docs/HOW_IT_WORKS.md#license)).

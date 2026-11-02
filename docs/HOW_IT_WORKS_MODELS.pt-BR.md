# Como funciona?

Voltar para [README.pt-BR.md](../README.pt-BR.md)

Modelos como este normalmente rodam em servidores com centenas de gigabytes de memória de vídeo. Sua placa de vídeo
tem 8-32 GB. O PolyStrata faz o modelo caber **dividindo o trabalho pelo PC inteiro**. Pense numa cozinha: o que você
usa o tempo todo fica na bancada, e o resto espera na despensa.

## Qwen3.8-Flash-Next

<p align="center"><img src="media/how-it-works.svg" width="860" alt="Os 24,576 experts do modelo: os mais usados na placa de vídeo, todos na RAM, uma tabela de consulta no SSD"></p>

- **O modelo é uma equipe de 24,576 pequenos especialistas ("experts").** Cada palavra precisa de só 10 deles.
- **Sua placa de vídeo** guarda os poucos milhares de experts usados com mais frequência. **Sua RAM** guarda todos
  eles, e **seu processador** trabalha no resto ao mesmo tempo. **Seu SSD** guarda uma grande tabela de consulta.

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="Um pequeno ajudante adivinha as próximas palavras; o modelo grande confere todas de uma vez e fica com as certas"></p>

- **Adivinhar e conferir:** um pequeno ajudante adivinha as próximas palavras. O modelo grande confere todas de
  uma vez. Você recebe a mesma resposta, 1.6-1.8x mais cedo.
- **Textos longos são lidos em grandes pedaços** (até 8,192 tokens por vez), a mais de 1,000 tokens por segundo.

A explicação mais longa: [docs/HOW_IT_WORKS.md](HOW_IT_WORKS.md). Cada parte e seus números:
[os detalhes](DETAILS.md#how-it-works) e o [artigo](paper/Strata-Paper.pdf).

## GLM 5.3 Flash

<p align="center"><img src="media/how-it-works-glm.svg" width="860" alt="Os 12,096 experts do modelo: os mais usados na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O modelo é uma equipe de 12,096 pequenos experts**, 288 em cada uma de 42 camadas. Cada palavra precisa de só 8 de
  cada camada.
- **Sua placa de vídeo** guarda os mais usados, cerca de 2,470 numa placa de 32 GB: o motor conta quais o seu trabalho
  usa e escolhe de novo na próxima inicialização. **Sua RAM** guarda os outros, e **seu processador** calcula esses ao
  mesmo tempo. **Seu SSD** guarda o arquivo do modelo, usado onde está, sem uma segunda cópia; uma inicialização leva
  cerca de 6 segundos quando o arquivo já está na sua RAM.

<p align="center"><img src="media/guess-and-check-glm.svg" width="860" alt="Um bloco do modelo adivinha as próximas palavras; o modelo confere todas de uma vez e fica com as certas"></p>

- **Adivinhar e conferir:** um bloco do modelo feito para adivinhar propõe as próximas palavras. O modelo confere
  todas de uma vez. Você recebe a mesma resposta, 1.1-1.2x mais cedo.
- **Textos longos são lidos em grandes pedaços** (1,024 tokens por vez; os experts que a placa não tem passam por
  ela), a 780 a 1,250 tokens por segundo.

Cada parte e seus números: [docs/GLM.md](GLM.md#how-it-uses-the-card) (em inglês).

## DeepSeek V4 Flash

<p align="center"><img src="media/how-it-works-dsv4.svg" width="860" alt="Os 11,008 experts do modelo: os mais usados na placa de vídeo, o resto na RAM, os arquivos do modelo no SSD"></p>

- **O modelo é uma equipe de 11,008 pequenos experts**, 256 em cada uma das suas 43 camadas. Cada palavra precisa só
  de 6 de cada camada, mais um que todas as palavras usam.
- **Sua placa de vídeo** guarda os mais usados, cerca de 2,840 numa placa de 32 GB: o motor conta quais o seu
  trabalho usa. **Sua RAM** guarda o resto, e **seu processador** os calcula ao mesmo tempo. **Seu SSD** guarda os
  arquivos do modelo, usados onde estão, sem uma segunda cópia.
- **Uma memória pequena da conversa:** uma palavra lê inteiras as 128 últimas palavras e o resto como linhas
  comprimidas, uma a cada 4 ou a cada 128 palavras. Um token de contexto ocupa 6,880 bytes, então um contexto longo
  deixa para a placa quase todo o espaço para os experts.
- **Textos longos são lidos em pedaços grandes** (1,024 tokens por vez), 30,000 tokens a cerca de 1,000 tokens por
  segundo. Ele ainda não tem "adivinhar e conferir": o modelo de rascunho feito para ele ainda não roda.

Cada parte e seus números: [docs/DEEPSEEK.md](DEEPSEEK.md#how-it-differs-from-the-other-models) (em inglês).

## LFM2.5-24B-A2B

<p align="center"><img src="media/how-it-works-lfm25.svg" width="860" alt="Os 2,432 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O modelo é uma equipe de 2,432 pequenos experts**, 64 em cada uma de 38 camadas. Cada palavra precisa de só 4 de
  cada camada.
- **A maioria das 40 camadas dele não tem atenção.** 30 delas misturam as palavras com uma convolução curta sobre as
  três últimas e guardam só dois vetores por camada. Só as outras 10 lembram a conversa inteira, então a memória dele
  de uma conversa é pequena: um contexto de 128K ocupa só 2 GB de VRAM a mais que 32K.
- **Sua placa de vídeo** guarda quantos experts couberem: cerca de 900 a 1,300 numa placa de 8 GB, e todos a partir
  de 24 GB nos tamanhos de 4 bits. Na placa de 8 GB os que ela guarda atendem de dois terços a quatro quintos do que
  uma resposta usa. **Sua RAM** guarda os outros, e **seu processador** calcula esses ao mesmo tempo. **Seu SSD**
  guarda o arquivo do modelo, usado onde está, sem uma segunda cópia.
- **Sem bloco de rascunho.** O modelo não tem nada que adivinhe o que vem a seguir, então o motor só adivinha a
  partir do texto que a conversa repete.
- **Quando a placa guarda todos os experts**, o passo inteiro fica na placa: ela mesma escolhe os experts das
  palavras e roda as 40 camadas de um token como um único lançamento gravado.

Cada parte e seus números: [docs/LFM.md](LFM.md) (em inglês).

## Flint 40B-A2.5B

<p align="center"><img src="media/how-it-works-flint.svg" width="860" alt="Os 12,032 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O modelo é uma equipe de 12,032 pequenos experts**, 256 em cada uma de 47 camadas. Cada palavra precisa de só 6 de
  cada camada, e de um expert compartilhado que toda palavra usa.
- **A maioria das 49 camadas dele olha 513 tokens para trás.** 35 delas só atendem aos últimos 513 tokens e guardam
  um anel de linhas que não cresce. Só as outras 14 lembram a conversa inteira, então a memória dele de uma conversa é
  pequena: um contexto de 128K ocupa só 2,6 GB de VRAM a mais que 32K.
- **Sua placa de vídeo** guarda quantos experts couberem: cerca de 1,600 a 2,500 numa placa de 8 GB, todos os do
  UD-Q4_K_XL numa placa de 32 GB. **Sua RAM** guarda os outros, e **seu processador** calcula esses ao mesmo tempo.
  **Seu SSD** guarda o arquivo do modelo, usado onde está, sem uma segunda cópia.
- **Sem bloco de rascunho.** O modelo não tem nada que adivinhe o que vem a seguir, então o motor só adivinha a
  partir do texto que a conversa repete.

Cada parte e seus números: [docs/FLINT.md](FLINT.md) (em inglês).

## Qwen3.6-35B-A3B

<p align="center"><img src="media/how-it-works-qwen36.svg" width="860" alt="Os 10,240 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O modelo é uma equipe de 10,240 pequenos experts**, 256 em cada uma de 40 camadas. Cada palavra precisa de só 8 de
  cada camada.
- **Sua placa de vídeo** guarda quantos couberem: cerca de 2,600 numa placa de 8 GB, todos a partir de 24 GB. Enquanto
  ele escreve, o motor traz para a placa os experts que a resposta escolhe sem parar, e na placa de 8 GB os que ela
  guarda atendem cerca de 60% do que uma resposta usa. **Sua RAM** guarda os outros, e **seu processador** calcula
  esses ao mesmo tempo. **Seu SSD** guarda o arquivo do modelo, usado onde está, sem uma segunda cópia.

<p align="center"><img src="media/guess-and-check-qwen36.svg" width="860" alt="O bloco de rascunho do modelo adivinha as próximas palavras; o modelo confere todas de uma vez e fica com as certas"></p>

- **Adivinhar e conferir:** o bloco de rascunho do próprio modelo, que está no arquivo dele, propõe as próximas
  palavras. O modelo confere todas de uma vez. Você recebe a mesma resposta, 1.2-1.5x mais cedo.
- **Textos longos são lidos em grandes pedaços** (1,024 tokens por vez). Quando a placa não guarda todos os experts,
  os outros passam por ela uma única vez para o prompt inteiro.

Cada parte e seus números: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (em inglês).

## Clover-35B-A3B-Preview

<p align="center"><img src="media/how-it-works-clover.svg" width="860" alt="Os 10,240 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O Qwen3.6-35B-A3B treinado mais:** uma equipe de 10,240 pequenos experts, 256 em cada uma de 40 camadas, e só 8 de
  cada camada para cada palavra.
- **Sua placa de vídeo** guarda quantos couberem: cerca de 2,200 a 4,100 numa placa de 8 GB, mais quanto menor o
  tamanho, e todos os do Q3_K_M a partir de 24 GB. Na placa de 8 GB os que ela guarda atendem de 60% a 80% do que uma
  resposta usa. **Sua RAM** guarda os outros, e **seu processador** calcula esses ao mesmo tempo. **Seu SSD** guarda o
  arquivo do modelo, usado onde está, sem uma segunda cópia.
- **Sem bloco de rascunho.** O Clover foi treinado para escrever um bloco de palavras de uma vez, preenchendo lacunas
  juntas, e assim também pode adivinhar as próximas palavras. Mas cada bloco adivinhado roda o modelo inteiro de novo,
  e isso custou mais do que os acertos economizaram, então está desligado
  ([medido](BACKENDS.md#clover-35b-a3b-preview), em inglês). O motor só adivinha a partir do texto que a conversa
  repete.
- **Textos longos são lidos em grandes pedaços** (1,024 tokens por vez), como no Qwen3.6-35B-A3B.

Cada parte e seus números: [docs/BACKENDS.md](BACKENDS.md#clover-35b-a3b-preview) (em inglês).

## Ornith 1.5

<p align="center"><img src="media/how-it-works-ornith.svg" width="860" alt="Os 10,240 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **A arquitetura do Qwen3.6-35B-A3B com outros pesos:** uma equipe de 10,240 pequenos experts, 256 em cada uma de 40
  camadas, e só 8 de cada camada para cada palavra.
- **Sua placa de vídeo** guarda quantos couberem: cerca de 3,100 numa placa de 8 GB, todos a partir de 24 GB. Enquanto
  ele escreve, o motor traz para a placa os experts que a resposta escolhe sem parar, e na placa de 8 GB os que ela
  guarda atendem cerca de 60% do que uma resposta usa. **Sua RAM** guarda os outros, e **seu processador** calcula
  esses ao mesmo tempo. **Seu SSD** guarda o arquivo do modelo, usado onde está, sem uma segunda cópia.

<p align="center"><img src="media/guess-and-check-ornith.svg" width="860" alt="O bloco de rascunho do modelo adivinha as próximas palavras; o modelo confere todas de uma vez e fica com as certas"></p>

- **Adivinhar e conferir:** o bloco de rascunho dele também está no arquivo e propõe as próximas palavras. Ele acerta
  menos vezes que o do Qwen3.6-35B-A3B, então você recebe a mesma resposta 1.1-1.2x mais cedo.
- **Textos longos são lidos em grandes pedaços** (1,024 tokens por vez). Quando a placa não guarda todos os experts,
  os outros passam por ela uma única vez para o prompt inteiro.

Cada parte e seus números: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (em inglês).

## Gemma 4 26B-A4B

<p align="center"><img src="media/how-it-works-gemma4.svg" width="860" alt="Os 3,840 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O modelo é uma equipe de 3,840 pequenos experts**, 128 em cada uma de 30 camadas. Cada palavra precisa de só 8 de
  cada camada, além de uma parte da camada que todas as palavras usam.
- **Sua placa de vídeo** guarda quantos couberem: cerca de 1,000 numa placa de 8 GB, todos a partir de 16 GB pela
  estimativa do instalador. Enquanto ele escreve, o motor traz para a placa os experts que a resposta escolhe sem
  parar, e na placa de 8 GB os que ela guarda atendem cerca de 55% do que uma resposta usa. **Sua RAM** guarda os
  outros, e **seu processador** calcula esses ao mesmo tempo. **Seu SSD** guarda o arquivo do modelo e o modelo de
  rascunho dele, usados onde estão, sem uma segunda cópia.

<p align="center"><img src="media/guess-and-check-gemma4.svg" width="860" alt="O pequeno modelo de rascunho do Google adivinha as próximas palavras; o modelo confere todas de uma vez e fica com as certas"></p>

- **Adivinhar e conferir:** o pequeno modelo de rascunho do Google para ele, um arquivo de 0.5 GB que o instalador
  baixa junto com o modelo, propõe as próximas palavras. O modelo confere todas de uma vez. Você recebe a mesma
  resposta, 1.3-1.6x mais cedo.
- **Textos longos são lidos em grandes pedaços** (1,024 tokens por vez). Quando a placa não guarda todos os experts,
  os outros passam por ela uma única vez para o prompt inteiro.

Cada parte e seus números: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (em inglês).

## Tiel Coder 35B-A3B

<p align="center"><img src="media/how-it-works-tiel.svg" width="860" alt="Os 10,240 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O Ornith 1.5 em outra quantização:** uma equipe de 10,240 pequenos experts, 256 em cada uma de 40
  camadas, e só 8 de cada camada para cada palavra.
- **Sua placa de vídeo** guarda quantos couberem: cerca de 2,700 numa placa de 8 GB, todos a partir de 24 GB. Enquanto
  ele escreve, o motor traz para a placa os experts que a resposta escolhe sem parar, e na placa de 8 GB os que ela
  guarda atendem cerca de 50% do que uma resposta usa. **Sua RAM** guarda os outros, e **seu processador** calcula
  esses ao mesmo tempo. **Seu SSD** guarda o arquivo do modelo, usado onde está, sem uma segunda cópia.

<p align="center"><img src="media/guess-and-check-tiel.svg" width="860" alt="O bloco de rascunho do modelo adivinha as próximas palavras; o modelo confere todas de uma vez e fica com as certas"></p>

- **Adivinhar e conferir:** o bloco de rascunho dele também está no arquivo e propõe as próximas palavras. Ele acerta
  menos vezes que o do Qwen3.6-35B-A3B, então você recebe a mesma resposta 1.2-1.3x mais cedo.
- **Textos longos são lidos em grandes pedaços** (1,024 tokens por vez). Quando a placa não guarda todos os experts,
  os outros passam por ela uma única vez para o prompt inteiro.

Cada parte e seus números: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (em inglês).

## MiniCPM-V 4.7 35B-A3B

<p align="center"><img src="media/how-it-works-minicpmv.svg" width="860" alt="Os 10,240 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **A arquitetura do Qwen3.6-35B-A3B com os pesos da OpenBMB:** uma equipe de 10,240 pequenos experts, 256 em cada
  uma de 40 camadas, e só 8 de cada camada para cada palavra.
- **Sua placa de vídeo** guarda quantos couberem: com Q4_K_M cerca de 2,000 numa placa de 8 GB, todos numa de 32 GB.
  Enquanto ele escreve, o motor traz para a placa os experts que a resposta escolhe sem parar, e na placa de 8 GB os
  que ela guarda atendem cerca de 60 a 70% do que uma resposta usa. **Sua RAM** guarda os outros, e **seu
  processador** calcula esses ao mesmo tempo. **Seu SSD** guarda o arquivo do modelo, usado onde está, sem uma
  segunda cópia.
- **Uma palavra por vez:** o arquivo dele não tem bloco de rascunho, então nada adivinha antes.
- **Imagens:** o codificador de imagens dele lê uma imagem grande como uma visão geral e alguns ladrilhos. Uma captura
  de 1,200 por 398 pixels vira cerca de 300 tokens, um gráfico de 2,520 por 1,620 cerca de 600.
- **Textos longos são lidos em grandes pedaços** (1,024 tokens por vez). Quando a placa não guarda todos os experts,
  os outros passam por ela uma única vez para o prompt inteiro.

Cada parte e seus números: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (em inglês).

## Kolibri 1

<p align="center"><img src="media/how-it-works-kolibri.svg" width="860" alt="Os 19,200 experts do modelo: os que uma resposta escolhe sem parar na placa de vídeo, os outros na RAM, o arquivo do modelo no SSD"></p>

- **O modelo é uma equipe de 19,200 pequenos experts**, 384 em cada uma de 50 camadas. Cada palavra precisa de só 6 de
  cada camada, e de um expert compartilhado que toda palavra usa.
- **Quatro de cada cinco camadas olham 513 tokens para trás.** Elas guardam um anel de linhas que não cresce. A
  quinta lembra a conversa inteira e não usa nenhuma posição das palavras.
- **Sua placa de vídeo** guarda quantos experts couberem: cerca de um décimo numa placa de 8 GB, todos os do Q2_K numa
  placa de 32 GB. **Sua RAM** guarda os outros, e **seu processador** calcula esses ao mesmo tempo. **Seu SSD** guarda
  o arquivo do modelo, usado onde está, sem uma segunda cópia.
- **Sem bloco de rascunho.** O modelo não tem nada que adivinhe o que vem a seguir, então o motor só adivinha a
  partir do texto que a conversa repete.

Cada parte e seus números: [docs/BACKENDS.md](BACKENDS.md#how-a-backend-works) (em inglês).

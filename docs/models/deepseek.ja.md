# DeepSeek V4 Flash

[README.ja.md](../../README.ja.md) に戻る

**DeepSeek V4 Flash**：ファイルは 91 GB で、これも大部分は RAM に置かれます。会話を覚えておくための VRAM が少なく、128K のコンテキストでも 32K より 0.7 GB 多いだけです。

## どのくらい速い？

PolyStrata が[GLM 5.3 Flash](glm.ja.md) と同じワークステーションで、短いチャットと 30K トークンのプロンプトで測った値です。

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-IQ2_XXS** | 56 tokens/s | 1,040 tokens/s |

</td></tr>
</table>

同じ PC で llama.cpp は毎秒 25〜28 トークンを書き出し、3,384 トークンのプロンプトを毎秒 96 トークンで読みました。
128K のコンテキストで 116,473 トークンのプロンプトを読んで答えるまで 174 秒、512K のコンテキストで 487,529 トークンの
プロンプトでは 32 分かかり、どちらも中に隠した 3 つの事実はすべて見つかりました。
測定の内容と方法：[docs/DEEPSEEK.md](../DEEPSEEK.md#what-it-was-measured-on)（英語）。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚。動かしたことがあるのは **VRAM 32 GB** だけです。エキスパートを載せる前に約 11.5 GB を使うので、12 GB のカードでは起動できません。 |
| **RAM** | 32 GB のカードと組み合わせる場合、**72 GB 以上**（インストーラーの見積もり）。カードが小さいほど RAM が多く必要です（[どのくらいか](#どのモデルを選べばいい)）。実際に動かしたのは 120 GB です。 |
| **ディスク** | モデルの 3 つのファイルで 91 GB。SSD に置いてください。 |
| **OS** | Linux と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。まだ試していないもの：AMD のカード、2 枚のカード。
読めるのはテキストだけです。Windows で 24 GB のカード 1 枚と 128 GB の RAM で動かしたユーザーの[報告](https://github.com/VecSzn/PolyStrata/issues/21)があります。

全体の一覧：[docs/DEEPSEEK.md](../DEEPSEEK.md#what-you-need)（英語）。

## どのモデルを選べばいい

いまのところ 1 サイズです。

| サイズ | 内容 | ダウンロード | 32 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **UD-IQ2_XXS** | モデル全体、約 2 ビット（[Unsloth](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)） | 91 GB | 72 GB |

GLM 5.3 Flash と同じく、カードに載らないエキスパートは RAM に置きます。インストーラーの見積もりでは、24 GB のカードなら約 80 GB、
16 GB のカードなら約 89 GB で、12 GB のカードでは起動しません。試したのは 32 GB のカードだけです。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#deepseek-v4-flash)。

# GLM 5.3 Flash

[README.ja.md](../../README.ja.md) に戻る

**GLM 5.3 Flash**：いちばん大きいモデル。ファイルは 79〜114 GB で、その大部分は RAM に置かれます。

## どのくらい速い？

PolyStrata が 1 台のワークステーションで、短いチャットと 29K トークンのプロンプトで測った値です。

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **3.0BIT** | 59-66 tokens/s | 820 tokens/s |
| **UD-IQ2_XXS** | 50-52 tokens/s | 1,060-1,080 tokens/s |
| **REAP50-Q3_K_M** | 59-66 tokens/s | 1,300-1,310 tokens/s |

</td></tr>
</table>

GLM 5.3 Flash はほかのモデルよりずっと大きく、このカードの 32 GB にはエキスパートの約 5 分の 1 しか載らないため、書き出しは遅くなります。
測定の内容と方法：[docs/GLM.md](../GLM.md#what-it-was-measured-on)（英語）。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚。動かしたことがあるのは **VRAM 32 GB** だけです。エキスパートを載せる前に約 12 GB を使うので、12 GB のカードでは起動できません。 |
| **RAM** | 32 GB のカードと組み合わせる場合、モデル全体で **96 GB 以上**、2 ビットのサイズで 82 GB、エキスパートを半分にしたサイズで 60 GB。カードが小さいほど RAM が多く必要です（[どのくらいか](#どのモデルを選べばいい)）。実際に動かしたのは 120 GB です。 |
| **ディスク** | モデルファイルに 114 GB、102 GB または 79 GB。SSD に置いてください。 |
| **OS** | Linux と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。エンジンはあなたの PC でコンパイルされ、足りないもの（g++ と CUDA ツールキット）は
インストーラーが入れるか尋ねます。未対応：AMD のカード、Windows での 2 枚、3 枚以上、画像。
Windows で 24 GB のカード 1 枚と 128 GB の RAM で動かしたユーザーの[報告](https://github.com/VecSzn/PolyStrata/issues/10)があります。

全体の一覧：[docs/GLM.md](../GLM.md#what-you-need)（英語）。

## どのモデルを選べばいい

いまのところ 3 サイズです。インストーラーは、RAM が足りる最初のサイズを選びます。

| サイズ | 内容 | ダウンロード | 32 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **3.0BIT** | モデル全体、約 3 ビット（[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)） | 114 GB | 96 GB |
| **UD-IQ2_XXS** | モデル全体、約 2 ビット（[Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)）。プロンプトの読み込みは速く、書き出しは遅くなります | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | ルーティングされるエキスパートを半分に減らしたもの（patrickbdevaney の [REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)） | 79 GB | 60 GB |

必要な RAM はグラフィックカードで決まります。カードに載らないエキスパートは RAM に置いておく必要があるからです。

| グラフィックカード | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | 実際に動かした PC（RAM 120 GB） |
| **24 GB** | 約 104 GB | 約 90 GB | 約 68 GB | 未実行 |
| **16 GB** | 約 113 GB | 約 99 GB | 約 77 GB | 未実行 |
| **12 GB 以下** | | | | 起動しません |

RAM の数字は、動かしたことのある 1 台の PC をもとにしたインストーラーの見積もりです。RAM が足りなくても起動はしますが、
回答中にエキスパートを SSD から読むことになり、ずっと遅くなります。インストーラーはそのことを伝え、先に確認します。

GLM 5.3 Flash のほかの量子化にはまだ対応していません：[足りないもの](../GLM.md#what-is-missing)（英語）。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#glm-53-flash)。

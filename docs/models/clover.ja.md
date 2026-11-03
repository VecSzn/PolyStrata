# Clover-35B-A3B-Preview

[README.ja.md](../../README.ja.md) に戻る

**Clover-35B-A3B-Preview**：DiffuSpace のモデル。ファイルは 13〜21 GB。Qwen3.6-35B-A3B をさらに学習させ、トークンのまとまりを一度に書けるようにしたものです。読めるのはテキストだけです。

## どのくらい速い？

PolyStrata がノート PC で、短いチャットと 3.2 万トークンのプロンプトで測りました。カードは 8 GB で、カードに載らないエキスパートはプロセッサーが計算します。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q3_K_M** | 68-73 tokens/s | 1,560-1,700 tokens/s |
| **APEX-I-NANOPLUS** | 72-76 tokens/s | 1,530-1,690 tokens/s |
| **Q4_K_M** | 38-58 tokens/s | 1,220-1,360 tokens/s |

</td></tr>
</table>

3.2 万トークンのプロンプトを読んだあと、Q3_K_M はノート PC で毎秒 60〜65 トークン書きます。Q4_K_M はそこで RAM 32 GB をいっぱいにするので、遅く、ばらつきます。RTX 5090 のワークステーションでは測っていません。

同じノート PC で Qwen3.6-35B-A3B のほうが速く書けるのは（毎秒 94〜100）、ファイルにドラフトブロックがあり、次の数語を 1 層分の計算で予想できるからです。Clover のファイルにはありません。Clover は自分でまとまりを予想することもできますが、それにはモデル全体をもう一度動かす必要があり、PolyStrata ではオフにしています。エキスパートが全部載るカードでも、予想しないほうが早く答えました（[測定](../BACKENDS.md#clover-35b-a3b-preview)、英語）。

Q3_K_M のエキスパートが全部載る RTX PRO 6000 で、PolyStrata が選んだ次のトークンは llama.cpp（そのプルリクエスト 31306 の `clover_moe` 対応）と、短いプロンプトのあと 128 ステップ中 127、5,573 トークンのあと 128 ステップ中 125 で同じでした。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 3.5 GB を使います。動かしたのはノート PC の 8 GB のカードと 96 GB のカードです。AMD のカードではまだ試していません。 |
| **RAM** | 8 GB のカードと組み合わせる場合、APEX-I-NANOPLUS、Q3_K_M、Q4_K_M で約 **16、20、25 GB**（インストーラーの見積もり）。24 GB のカードなら、小さいほうの 2 サイズはモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 13、17、21 GB のいずれか。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはインストーラーが用意します。[Qwen3.6-35B-A3B](qwen36.ja.md) と同じです。maternion のファイルにはチャットテンプレートがないので、インストーラーがモデル自身のテンプレートを DiffuSpace からダウンロードします。Clover のライセンスは Apache 2.0 です。

## どのモデルを選べばいい

3 サイズで、どれも llama.cpp のプルリクエスト 31306 の変換スクリプトで作られています。インストーラーは Q3_K_M を選び、RAM が足りない PC では APEX-I-NANOPLUS を選びます。Q4_K_M は名前で指定します（`--model Q4_K_M`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **Q3_K_M** | モデル全体、エキスパートは 3 ビット（[maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)） | 17 GB | 約 20 GB |
| **APEX-I-NANOPLUS** | エキスパートは 2 ビットと 3 ビット、重要度行列で量子化（[IsValorum](https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF)）。必要な RAM が最も少ない | 13 GB | 約 16 GB |
| **Q4_K_M** | エキスパートは 4 ビット（maternion） | 21 GB | 約 25 GB |

maternion の 2 サイズは重要度行列なしで量子化されています。各サイズが元のモデルにどれだけ近いかは測っていません。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#clover-35b-a3b-preview)。

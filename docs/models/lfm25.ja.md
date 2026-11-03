# LFM2.5-24B-A2B

[README.ja.md](../../README.ja.md) に戻る

**LFM2.5-24B-A2B**：Liquid AI のモデルで、ファイルは 11〜25 GB。ほとんどの層にアテンションがないので、長い会話でも VRAM をあまり使いません。128K のコンテキストでも 32K より 2 GB 多いだけです。ライセンスで商用利用が認められるのは、年間売上が 1,000 万ドル未満の会社だけです。

## どのくらい速い？

[Qwen3.6-35B-A3B](qwen36.ja.md) と同じ測り方で、同じ 2 台の PC、約 3 万トークンのプロンプトで測りました。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、RAM 120 GB</th></tr>
<tr><td>

| サイズ | 回答を書く | プロンプトを読む |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 70-73 tokens/s | 3,590-3,620 tokens/s |
| **UD-Q3_K_XL** | 88 tokens/s | 3,760-3,810 tokens/s |
| **QAD-Q4_0** | 77-78 tokens/s | 3,720-3,730 tokens/s |
| **UD-Q6_K_XL** | 42-43 tokens/s | 3,030-3,120 tokens/s |
| **Q8_0** | 23-34 tokens/s | 2,950-3,040 tokens/s |

</td><td>

| サイズ | 回答を書く | プロンプトを読む |
| --- | ---: | ---: |
| **UD-Q4_K_XL** | 480-485 tokens/s | 23,390-23,460 tokens/s |
| **UD-Q3_K_XL** | 490-497 tokens/s | 22,920-23,400 tokens/s |
| **QAD-Q4_0** | 498 tokens/s | 23,880-23,900 tokens/s |
| **UD-Q6_K_XL** | 423-426 tokens/s | 21,250-21,270 tokens/s |
| **Q8_0** | 376 tokens/s | 23,270-23,350 tokens/s |

</td></tr>
</table>

ノート PC では Q8_0 が 32 GB の RAM をいっぱいに使うので遅くなります。ワークステーションにはどのサイズもモデル全体が載ります。

ノート PC では、そこで試した設定でいちばん速かった `--n-cpu-moe 25` の llama.cpp b11540 が 32,000 トークンのプロンプトを 530 トークン/秒で読み、そのあと 53 で書きました。
PolyStrata は 3,540 で読み、71〜73 で書きました。
ワークステーションでは llama-bench が 29,816 トークンを llama.cpp の既定のバッチ 512 で 13,300 トークン/秒、2,048 で 23,600 と PolyStrata と同じ速さで読み、1 トークンずつ 455 トークン/秒、そのプロンプトのあとは 369 で書きました。
PolyStrata は 452 と 387 です。
128K のコンテキストで約 124,000 トークンのプロンプトを、ワークステーションは 8.7 秒で読んで答え、ノート PC は 45 秒で読みました。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 4 GB を使います。動かしたのはノート PC の 8 GB のカードと 32 GB のカードです。 |
| **RAM** | 8 GB のカードと組み合わせる場合、UD-Q3_K_XL、QAD-Q4_0、UD-Q4_K_XL、UD-Q6_K_XL、Q8_0 でそれぞれ約 **16、18、20、27、31 GB**（インストーラーの見積もり）。24 GB のカードなら、小さいほうの 3 サイズはモデル全体が載ります。5 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 11〜25 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。[Qwen3.6-35B-A3B](qwen36.ja.md) と同じです。ユーザーが Linux の AMD RX 6700 XT（12 GB）で動かしています。テキストだけを読みます。
Liquid AI のライセンスで商用利用が認められるのは、年間売上が 1,000 万ドル未満の会社だけです（[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)）。

詳しい一覧：[docs/LFM.md](../LFM.md#what-you-need)（英語）。

## どのモデルを選べばいい

5 サイズです。インストーラーは UD-Q4_K_XL を選び、RAM が足りない PC では UD-Q3_K_XL を選びます。ほかは名前で指定します（`--model QAD-Q4_0`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | モデル全体、エキスパートは 4 ビット（[Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)） | 15 GB | 約 20 GB |
| **UD-Q3_K_XL** | 同じモデル、エキスパートは 3 ビット。必要な RAM が最も少ない | 11 GB | 約 16 GB |
| **QAD-Q4_0** | 4 ビット。[Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) 自身のファイルで、この量子化で失うものが少なくなるように訓練されています | 14 GB | 約 18 GB |
| **UD-Q6_K_XL** | エキスパートは 6 ビット | 22 GB | 約 27 GB |
| **Q8_0** | すべて 8 ビット。訓練されたままのモデルにいちばん近い | 25 GB | 約 31 GB |

Q8_0 と比べると、256 トークンずつの 5 つのプロンプトで、ほかのサイズが Q8_0 と同じいちばん確率の高いトークンを選んだ
ステップは 96.8%（UD-Q6_K_XL）、95.5%（UD-Q4_K_XL）、92.7%（QAD-Q4_0）、91.6%（UD-Q3_K_XL）でした。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#lfm25-24b-a2b)。

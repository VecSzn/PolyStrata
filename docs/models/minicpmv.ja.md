# MiniCPM-V 4.7 35B-A3B

[README.ja.md](../../README.ja.md) に戻る

**MiniCPM-V 4.7 35B-A3B**：OpenBMB のモデル。Qwen3.6-35B-A3B と同じ構造で、重みは独自のもの。画像を読むために作られています。ファイルは 16〜29 GB。

## どのくらい速い？

[Qwen3.6-35B-A3B](qwen36.ja.md) と同じ方法、同じ 2 台の PC で測りました。ファイルにドラフトブロックがないので、1 トークンずつ書きます。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q4_K_M** | 64 tokens/s | 1,150 tokens/s |
| **mix_q3gud_im** | 59 tokens/s | 1,130-1,150 tokens/s |
| **Q6_K** | 40 tokens/s | 960-980 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q4_K_M** | 308-311 tokens/s | 7,910-7,950 tokens/s |
| **mix_q3gud_im** | 291-293 tokens/s | 7,730-7,740 tokens/s |
| **Q6_K** | 277-278 tokens/s | 7,290-7,330 tokens/s |

</td></tr>
</table>

10,000 トークンのプロンプトは、ノート PC で毎秒 1,260 トークン、ワークステーションで毎秒 8,790 トークンで読みます。
ノート PC では Q6_K が 32 GB の RAM を使い切るので、そこでは遅くなります。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 3.5 GB を使います。動かしたのは 32 GB のカードと、ノート PC の 8 GB のカードです。 AMD のカードでは、Linux と Windows でベータとして動きます。セットアップがこの PC でエンジンをコンパイルします。RX 7600 で動きました。 |
| **RAM** | 8 GB のカードと組み合わせる場合、3 つのサイズで約 **19、25、31 GB**（インストーラーの見積もり）。32 GB のカードなら、3 つのサイズのどれもモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 16、21、29 GB のいずれか。画像を読ませるならさらに 1 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。
Windows では最新リリースからビルド済みのエンジンをダウンロードします。
AVX2 のない CPU と、そのリリースのあとでエンジンが変わったコピーでは、代わりに Visual Studio の 2022 コンパイラーと
CUDA ツールキットでコンパイルし、足りなければインストールするか尋ねます。未対応：Windows での 2 枚、3 枚以上。Windows の AMD のカードでは、セットアップが Linux と同じように `strata-poly` をコンパイルします（AMD の TheRock ホイールを `.rocm-win` に）。そこでの実行は新しいものだという警告が出ます（[AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)）。

## どのモデルを選べばいい

いまのところ 3 サイズで、OpenBMB のものです。インストーラーは Q4_K_M を選び、RAM が足りない PC では mix_q3gud_im を選びます。Q6_K は名前で指定します（`--model Q6_K`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **Q4_K_M** | モデル全体、エキスパートは 4 ビット（[OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)） | 21 GB | 約 25 GB |
| **mix_q3gud_im** | 同じモデル、エキスパートは 3 ビットと 4 ビット。必要な RAM が最も少ない | 16 GB | 約 19 GB |
| **Q6_K** | 同じモデル、エキスパートは 6 ビット | 29 GB | 約 31 GB |

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#minicpm-v-47-35b-a3b)。

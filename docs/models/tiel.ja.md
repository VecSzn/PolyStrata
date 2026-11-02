# Tiel Coder 35B-A3B

[README.ja.md](../../README.ja.md) に戻る

**Tiel Coder 35B-A3B**：peculiar-ragdoll がコーディングエージェント向けに量子化した Ornith 1.5。ファイルは 14〜23 GB。画像も読めます。

## どのくらい速い？

[Qwen3.6-35B-A3B](qwen36.ja.md) と同じ方法、同じ 2 台の PC で測りました。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 83 tokens/s | 1,700 tokens/s |
| **UD-IQ3_XXS** | 96-97 tokens/s | 1,610 tokens/s |
| **UD-Q4_K_XL** | 74-79 tokens/s | 1,660-1,670 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_XL** | 337-345 tokens/s | 8,080-8,110 tokens/s |
| **UD-IQ3_XXS** | 352 tokens/s | 7,520 tokens/s |
| **UD-Q4_K_XL** | 316-322 tokens/s | 7,750-7,970 tokens/s |

</td></tr>
</table>

10,000 トークンのプロンプトは、ノート PC で毎秒 1,670 トークン、ワークステーションで毎秒 6,300 トークンで読みます。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 5 GB を使います。動かしたのは 32 GB のカード、そのカードを 8 GB に制限した状態、ノート PC の 8 GB のカードです。 AMD のカードでは、Linux と Windows でベータとして動きます。セットアップがこの PC でエンジンをコンパイルします。ユーザーが Linux の RX 6700 XT と RX 7800 XT、Windows の RX 9070 XT で Tiel Coder を動かしています。 |
| **RAM** | 8 GB のカードと組み合わせる場合、3 つのサイズで約 **18、21、27 GB**（インストーラーの見積もり）。24 GB のカードなら、3 つのサイズのどれもモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 14、17、23 GB のいずれか。画像を読ませるならさらに 1 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。
Windows では最新リリースからビルド済みのエンジンをダウンロードします。
AVX2 のない CPU と、そのリリースのあとでエンジンが変わったコピーでは、代わりに Visual Studio の 2022 コンパイラーと
CUDA ツールキットでコンパイルし、足りなければインストールするか尋ねます。未対応：Windows での 2 枚、3 枚以上。Windows の AMD のカードでは、セットアップが Linux と同じように `strata-poly` をコンパイルします（AMD の TheRock ホイールを `.rocm-win` に）。そこでの実行は新しいものだという警告が出ます（[AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)）。

## どのモデルを選べばいい

いまのところ 3 サイズです。インストーラーは UD-Q3_K_XL を選び、RAM が足りない PC では UD-IQ3_XXS を選びます。UD-Q4_K_XL は名前で指定します（`--model UD-Q4_K_XL`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **UD-Q3_K_XL** | モデル全体、エキスパートは 3 ビットと 4 ビット（[peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)） | 17 GB | 約 21 GB |
| **UD-IQ3_XXS** | 同じモデル、エキスパートは 2 ビットと 3 ビット。必要な RAM が最も少ない | 14 GB | 約 18 GB |
| **UD-Q4_K_XL** | 同じモデル、エキスパートは 4〜6 ビット | 23 GB | 約 27 GB |

カードが大きいほどモデルを多く載せられ、必要な RAM は減ります。UD-Q3_K_XL を 12 GB のカードと組む場合は約 17 GB、16 GB のカードなら 13 GB、カードにすべて載れば 10 GB です。
これらはインストーラーの見積もりで、そのような PC では確かめていません。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#tiel-coder-35b-a3b)。

# Qwen3.6-35B-A3B

[README.ja.md](../../README.ja.md) に戻る

**Qwen3.6-35B-A3B**：ファイルは 13〜23 GB。画像も読めます。

## どのくらい速い？

PolyStrata が 2 台の PC で、短いチャットと 3 万〜3.2 万トークンのプロンプトで測りました。左は 8 GB のカードを載せたノート PC で、
カードに載らないエキスパートはプロセッサーが計算します。右はカードにモデル全体が載るワークステーションです。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 94-100 tokens/s | 1,690 tokens/s |
| **UD-Q2_K_XL** | 123-124 tokens/s | 1,590 tokens/s |
| **UD-Q4_K_M** | 82-86 tokens/s | 1,560-1,570 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 394-395 tokens/s | 7,800-7,810 tokens/s |
| **UD-Q2_K_XL** | 450 tokens/s | 7,720-7,740 tokens/s |
| **UD-Q4_K_M** | 395-396 tokens/s | 8,070-8,080 tokens/s |

</td></tr>
</table>

10,000 トークンのプロンプトは、ノート PC で毎秒 1,660 トークン、
ワークステーションで毎秒 6,300 トークンで読みます。
ワークステーションのカードを 8 GB に制限した数字は、ノート PC を測る前の代わりの値で、
[docs/BACKENDS.md](../BACKENDS.md#with-8-gb-of-vram)（英語）にあります。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 5 GB を使います。動かしたのは 32 GB のカード、そのカードを 8 GB に制限した状態、ノート PC の 8 GB のカードです。 AMD のカードでは、Linux と Windows でベータとして動きます。セットアップがこの PC でエンジンをコンパイルします。これまでに RX 7600 で Qwen3.6 と Ornith を試しただけです。 |
| **RAM** | 8 GB のカードと組み合わせる場合、3 つのサイズで約 **17、21、27 GB**（インストーラーの見積もり）。24 GB のカードなら、小さいほうの 2 サイズはモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 13、17、23 GB のいずれか。画像を読ませるならさらに 1 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。
Windows では最新リリースからビルド済みのエンジンをダウンロードします。
AVX2 のない CPU と、そのリリースのあとでエンジンが変わったコピーでは、代わりに Visual Studio の 2022 コンパイラーと
CUDA ツールキットでコンパイルし、足りなければインストールするか尋ねます。未対応：Windows での 2 枚、3 枚以上。Windows の AMD のカードでは、セットアップが Linux と同じように `strata-poly` をコンパイルします（AMD の TheRock ホイールを `.rocm-win` に）。そこでの実行は新しいものだという警告が出ます（[AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)）。

[Strata](https://github.com/Niko1221/Strata) 自身のエンジンも 0.1.44 から Qwen3.6 を動かせます。UD-Q4_K_XL 以上です（`--family qwen36-strata`、RAM 32 GB 以上）。
RTX 5090 と UD-Q4_K_XL で比べると、4K のプロンプトは strata-poly の 1.6 倍の速さで読み、31K のプロンプトと書き出しはほぼ同じでした（[測定](../BACKENDS.md#qwen36-on-two-engines)、英語）。
小さいサイズと 8 GB のカードは strata-poly のままです。

## どのモデルを選べばいい

いまのところ 3 サイズです。インストーラーは UD-Q3_K_M を選び、RAM が足りない PC では UD-Q2_K_XL を選びます。UD-Q4_K_M は名前で指定します（`--model UD-Q4_K_M`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | モデル全体、エキスパートは 3 ビット（[Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)） | 17 GB | 約 21 GB |
| **UD-Q2_K_XL** | 同じモデル、エキスパートは 2 ビットと 3 ビット。必要な RAM が最も少ない | 13 GB | 約 17 GB |
| **UD-Q4_K_M** | 同じモデル、エキスパートは 4 ビット | 23 GB | 約 27 GB |

カードが大きいほどモデルを多く載せられ、必要な RAM は減ります。UD-Q3_K_M を 12 GB のカードと組む場合は約 17 GB、16 GB のカードなら 13 GB、カードにすべて載れば 10 GB です。
これらはインストーラーの見積もりで、そのような PC では確かめていません。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#qwen36-35b-a3b)。

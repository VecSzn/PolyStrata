# Ornith 1.5

[README.ja.md](../../README.ja.md) に戻る

**Ornith 1.5**：構造は Qwen3.6-35B-A3B と同じで、重みが違います。ファイルは 14〜24 GB。画像も読めます。

## どのくらい速い？

[Qwen3.6-35B-A3B](qwen36.ja.md) と同じ方法、同じ 2 台の PC で測りました。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 76-82 tokens/s | 1,620 tokens/s |
| **APEX-I-MINI** | 80-86 tokens/s | 1,550-1,560 tokens/s |
| **APEX-I-QUALITY** | 59-63 tokens/s | 1,570-1,580 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 316-319 tokens/s | 8,040-8,110 tokens/s |
| **APEX-I-MINI** | 289-300 tokens/s | 7,710-7,740 tokens/s |
| **APEX-I-QUALITY** | 297-302 tokens/s | 7,860-7,880 tokens/s |

</td></tr>
</table>

10,000 トークンのプロンプトは、ノート PC で毎秒 1,590 トークン、ワークステーションで毎秒 6,400 トークンで読みます。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 4 GB を使います。動かしたのは 32 GB のカード、そのカードを 8 GB に制限した状態、ノート PC の 8 GB のカードです。 AMD のカードでは、Linux と Windows でベータとして動きます。セットアップがこの PC でエンジンをコンパイルします。これまでに RX 7600 で Qwen3.6 と Ornith を試しただけです。 |
| **RAM** | 8 GB のカードと組み合わせる場合、3 つのサイズで約 **18、22、28 GB**（インストーラーの見積もり）。24 GB のカードなら、小さいほうの 2 サイズはモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 14、17、24 GB のいずれか。画像を読ませるならさらに 1 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。
Windows では最新リリースからビルド済みのエンジンをダウンロードします。
AVX2 のない CPU と、そのリリースのあとでエンジンが変わったコピーでは、代わりに Visual Studio の 2022 コンパイラーと
CUDA ツールキットでコンパイルし、足りなければインストールするか尋ねます。未対応：Windows での 2 枚、3 枚以上。Windows の AMD のカードでは、セットアップが Linux と同じように `strata-poly` をコンパイルします（AMD の TheRock ホイールを `.rocm-win` に）。そこでの実行は新しいものだという警告が出ます（[AMD_HIP.md](../AMD_HIP.md#strata-poly-on-amd-beta)）。

## どのモデルを選べばいい

いまのところ 3 サイズです。インストーラーは APEX-I-COMPACT を選び、RAM が足りない PC では APEX-I-MINI を選びます。APEX-I-QUALITY は名前で指定します（`--model APEX-I-QUALITY`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | モデル全体、エキスパートは 3 ビットと 4 ビット（mudler の [APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF)） | 17 GB | 約 22 GB |
| **APEX-I-MINI** | 同じモデル、エキスパートは 2 ビットと 3 ビット。必要な RAM が最も少ない | 14 GB | 約 18 GB |
| **APEX-I-QUALITY** | 同じモデル、エキスパートは 4〜6 ビット | 24 GB | 約 28 GB |

カードが大きいほどモデルを多く載せられ、必要な RAM は減ります。APEX-I-COMPACT を 12 GB のカードと組む場合は約 17 GB、16 GB のカードなら 13 GB、カードにすべて載れば 10 GB です。
これらはインストーラーの見積もりで、そのような PC では確かめていません。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#ornith-15)。

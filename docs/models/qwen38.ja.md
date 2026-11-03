# Qwen3.8-Flash-Next

[README.ja.md](../../README.ja.md) に戻る

**Qwen3.8-Flash-Next**：12 GB 以上の NVIDIA または AMD のグラフィックカード、Windows または Linux。画像も読めます。エンジンは[Strata](https://github.com/Niko1221/Strata) のものをそのまま使っています。

## どのくらい速い？

[Strata](https://github.com/Niko1221/Strata) がごく普通のゲーミング PC 2 台で測った値です。プロンプトは 32K トークン。

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA：Q2_0 はエンジン 0.1.36、ほかの行は 0.1.26 で測定（回答 4K、プロンプト 32K）。表の全体は
[DETAILS.md](../DETAILS.md#speed-measured) にあります。VRAM が多いカードほど速くなります。RTX 3090（24 GB）なら
毎秒およそ 100-140 トークンで書き出せるはずです。長いチャットやほかのカードについて：[モデルごとの速度](../MODELS.md#how-fast-is-each-size)、
[コミュニティの測定結果](../COMMUNITY_BENCHMARKS.md)。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20、30、40、50 シリーズ、または **AMD** Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700、RX 6800 / 6900 シリーズ。**VRAM 12 GB 以上**が必要です。 |
| **RAM** | 32 GB 以上。RAM の量で[使えるモデル](#どのモデルを選べばいい)が決まります。64 GB あればすべてのサイズが動きます。 |
| **ディスク** | 空き容量が約 80 GB。できれば SSD を使ってください。初回の起動がずっと速くなります。 |
| **OS** | Windows 10 / 11 または Linux。NVIDIA または AMD の最新のグラフィックドライバー。 |

ほかに必要なものはすべてインストーラーが用意します。2 枚や 3 枚のカードでモデルを分担することもできます（[マルチ GPU](../MULTI_GPU.md)）。

試験的なサポート。コミュニティのメンバーが自分のマシンで書き、テストしたものです：

- **古いグラフィックカード**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[Older GPUs](../OLDER_GPUS.md)。
- **Intel Arc**（Linux でソースからビルド）：[Intel Arc](../INTEL_ARC.md)。
- **AMD Ryzen AI Max（Strix Halo）**（Linux でソースからビルド）：[Strix Halo](../STRIX_HALO.md)。
- **AVX2 のない古いプロセッサー**：動きますが、遅いです。[Older CPUs](../INSTALL.md#older-cpus-experimental)。

全リスト：[docs/INSTALL.md](../INSTALL.md#what-you-need)。

## どのモデルを選べばいい

インストーラーが RAM に合わせてひとつ勧めます。同じモデルにも、圧縮の強さが違う複数のサイズがあります。
小さいサイズほど速く、大きいサイズほど少し賢くなります。

| RAM | 選ぶもの | 理由 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB に収まり、コード向けに作られている（24 GB のカードなら Q2_0 と IQ2_XS も動く） |
| **48 GB** | **IQ2_XS**（または最速の Q2_0） | 大きいサイズは収まらない |
| **64 GB** | **IQ2_XS**（おすすめ）、または IQ3_XXS / IQ3_S | すべてのサイズが収まる。IQ3_S がいちばん賢く、いちばん遅い |
| **96 GB 以上** | **IQ3_S**、または Unsloth の UD-IQ4_XS（約 4-bit） | ほかのものを全部開いたままでも、最大のサイズが入る余裕がある |

- **[Coder](../MODELS.md#coder)：** エキスパートを半分取り除いたコーディング向け版です。フルモデルの SWE-bench Verified スコアの
  91% に届き（作者による測定）、32 GB の RAM に収まります。コード以外では弱く、中国語などの CJK のテキストも苦手です（#438）。
  そうした用途には、すべてのエキスパートを残している Q2_0、IQ2_XS、IQ3_S を選んでください。
- **[Swift 1.5](../MODELS.md#swift-15)：** 答える前に考える時間がずっと短いファインチューン版です。
  ほぼ同じ品質のまま、答えが早く返ってきます。
- **[Unsloth UD-IQ4_XS](../MODELS.md#unsloth-ud-iq4_xs)：** Unsloth の約 4-bit 版で、品質は IQ3_S と UD-Q4_K_XL の間です。
  ダウンロードは 94 GB。RAM が約 80 GB 未満だと、答えている間に PolyStrata がその一部を SSD から読むので遅くなります（NVMe SSD が有効です）。
- **[Unsloth UD-Q4_K_XL / UD-Q5_K_XL / UD-Q6_K_XL / Q8_0](../MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0)：** 4〜8 ビットで、フルモデルにより近いモデルです
  （次のトークンが Q8_0 と一致する割合は 93〜95%）。すべてのエキスパートが RAM に収まる場合（UD-Q4_K_XL で約 88 GB、
  Q8_0 で約 136 GB）は高速です。RAM が足りない場合は回答の生成中に一部を SSD から読み込むため、大幅に遅くなります
  （RAM が 64 GB の PC で UD-Q4_K_XL は毎秒 7-8.5 トークン程度）。
- **[OrcaRouter's Uncensored IQ3_XXS](../MODELS.md#orcarouter-uncensored-iq3_xxs)：** 手動でセットアップします。
  インストーラーのメニューにはありません。

サイズ、ダウンロード、どれがどこに収まるかについて：[docs/MODELS.md](../MODELS.md)。あとで別のモデルを追加するには
`SETUP.bat`（Linux：`./setup.sh --setup`）を実行します。

## しくみ

図と説明：[HOW_IT_WORKS_MODELS.ja.md](../HOW_IT_WORKS_MODELS.ja.md#qwen38-flash-next)。

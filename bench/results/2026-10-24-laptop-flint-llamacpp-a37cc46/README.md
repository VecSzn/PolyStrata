# Flint 40B-A2.5B UD-Q3_K_XL on the 8 GB laptop: PolyStrata and llama.cpp a37cc46104 (2026-10-24)

RTX 4070 Laptop (8 GB), i7-14700HX, 32 GB of RAM, Windows 10. `bench/bench_vs_llama.py run --machine
machine-laptop.json --rounds 2`: each server in turn, a warm-up, then the prompts of the LFM2.5 run of 2026-10-15
(`prompt-4k.txt`, `prompt-32k.txt`: 3,566 and 29,442 of Flint's tokens) with a 36,864 context, at temperature 0 and
0.7. `summary.txt` is `bench_vs_llama.py summary` over the folder; `raw/` has every request.

UD-Q3_K_XL is the size setup takes on this laptop. llama.cpp is a37cc46104, the first commit with Flint, built for
CUDA on the laptop.

| Config | Engine | Settings |
| --- | --- | --- |
| `polystrata-setup` | strata-poly of 3863358e | the run config setup writes for this laptop (`strata-cfg-polystrata-setup.json`) |
| `llama-cpumoe36` | llama-server a37cc46104 | `-ngl 99 --n-cpu-moe 36 -fa on -ctk q8_0 -ctv q8_0 -t 20 -ub 512` |
| `llama-cpumoe40` | llama-server a37cc46104 | the same with `--n-cpu-moe 40` |
| `llama-fit` | llama-server a37cc46104 | `--fit on -fa on -ctk q8_0 -ctv q8_0`, the fastest of the three |

| | PolyStrata | llama.cpp, `llama-fit` | llama.cpp, `--n-cpu-moe 36` |
| --- | ---: | ---: | ---: |
| Reads 3,566 tokens | 1,682 to 1,773 tokens/s | 402 to 406 | 392 to 409 |
| Writes after them | 63 to 69 tokens/s | 42.3 to 43.2 | 39.6 to 43.9 |
| Reads 29,442 tokens | 1,946 to 2,012 tokens/s | 390 to 391 | 379 to 394 |
| Writes after them | 49 to 52 tokens/s | 35.3 to 35.4 | 31.9 to 35.6 |
| First token after 29,442 | 14.6 to 15.1 s | 75.3 to 75.4 s | 74.7 to 77.7 s |

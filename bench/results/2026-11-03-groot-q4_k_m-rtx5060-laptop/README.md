# Groot Q4_K_M rerun on an RTX 5060 Laptop 8 GB (2026-11-03)

Same-day rerun of the 2026-11-02 benchmark on branch `add-groot-family` (commit `ca522caa`, whose tree is identical
to `8ab2be06`), to check that the numbers in [the model page](../../../docs/models/groot.md) reproduce. Engine:
ready-made `strata-poly` 0.1.45 (CUDA 13.0), the same build as the first run. Model, PC and settings are those of
the 2026-11-02 run: Q4_K_M, 32,768-token context, MTP on, temperature 0, 256-token answers, Core Ultra 9 275HX
(15 threads), 32 GB RAM, driver 617.42, Windows 11.

`python polystrata.py benchmark --model groot-19b-a3b-261023-q4_k_m` was run twice: once as it comes, and once with
one small warm-up request first, because the command has no warm-up of its own and the first request pays for it.

| Run | Prompt, tokens | Prefill, tokens/s | Decode, tokens/s | MTP kept | Peak VRAM | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | 21 | 6.4 | 93.8 | 82.9% | 7.09 GiB | 21.29 GiB |
| cold | 2,952 | 1,446.2 | 99.0 | 82.8% | 7.18 GiB | 19.74 GiB |
| cold | 10,649 | 1,886.4 | 101.5 | 79.4% | 7.18 GiB | 19.74 GiB |
| warm | 21 | 122.7 | 83.9 | 66.5% | 7.09 GiB | 19.35 GiB |
| warm | 2,952 | 1,657.7 | 96.5 | 79.4% | 7.17 GiB | 18.76 GiB |
| warm | 10,649 | 1,928.9 | 101.3 | 79.9% | 7.17 GiB | 18.25 GiB |

The 2026-11-02 table was 74.6 / 96.9 / 95.4 decode and 74.6, 1,296.4, 1,904.8 prefill. Both reruns land around it:

- **Decode** is 93.8-101.5 tok/s in the cold run and 83.9-101.3 in the warm one; the long prompts stay within 7% of
  the first run in both.
- **Prefill** is 1,446-1,929 tok/s on the long prompts, against 1,296-1,905 on 2026-11-02; the 2,952-token prompt is
  the largest difference (+12% cold, +28% warm).
- **The 21-token prompt swings the most.** Its prefill is 6.4 tok/s cold (3.3 s for 21 tokens is start-up work, not
  throughput) and 122.7 tok/s after a warm-up. Its decode follows the answer the model happens to write, because the
  MTP acceptance differs between runs (66.5% warm vs 82.9% cold).
- **VRAM** peaked at 7.09-7.18 GiB whole machine, against 6.84-7.05 GiB on 2026-11-02 — about 0.1 GiB more, inside
  the run-to-run spread of the card's other users (the display).
- **RAM** peaked at 18.3-21.3 GiB whole machine, against 21.5-21.8 GiB. The figure counts every process on the PC,
  so the difference is what else was open, not the engine.

The warm run also recorded the warm-up row itself: 13 tokens, 33.1 prefill and 48.9 decode tok/s, which shows how
much of the first request is set-up.

Raw rows: `benchmark-cold.json` and `benchmark-warm.json`, in the same shape as the 2026-11-02 `benchmark.json`.

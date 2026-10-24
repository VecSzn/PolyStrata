# Ornith 122B GQA-16 attention validation

Measured 2026-10-21 on one RTX 5060 Ti by ersensari. This validates the optional attention kernels; CPU KV
residency and expert placement are unchanged.

## Hardware and software

- RTX 5060 Ti, 16,311 MiB reported VRAM, 180 W power limit; only host GPU 1 visible in the test container.
  The other card was not used. Maximum PCIe link reported as generation 5, x8; negotiated transfer rate not measured.
- Ryzen 9 9900X, 12 cores / 24 threads; 91 GiB RAM reported by Linux; BIWIN NV7400 NVMe model storage.
- Linux 7.0.0-38-generic, NVIDIA driver 595.84, CUDA 13.0.48, sm_120 source build.
- Source base: `c7916cf5c4a91456384b3fa28879c6949d9480d2`. Default-output reference: official
  `v0.1.45-poly.2` engine (source `974552f562aecb3e86cdd2679e76e4515b52b9d2`).
- The production inference container was stopped during GPU tests. Other host services remained running.
  GPU 0 had a separate idle model allocation. Page cache was not dropped; no host memory or power peak measured.

## Build and kernel tests

The container had CUDA and PolyStrata's pinned, patched llama.cpp sources at `LLAMA_SOURCE`:

```sh
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSTRATA_GGML_DIR="$LLAMA_SOURCE"
cmake --build build --target poly_fattn_test strata-poly -j8
build/poly_fattn_test gqa16
build/poly_fattn_test dec gqa16
build/poly_fattn_test
build/poly_fattn_test dec
```

All 92 cases passed: 20 prompt and 26 decode cases for each group, including Q8 and FP16 caches, causal tile
boundaries, small prompt scratch, and decode launches spanning 9 tokens. Decode checks compare each token alone,
in a window and using the position on the card, bit for bit. Numerical tolerances were not changed. Prompt
references include three tokens calculated in doubles; decode includes first/last token heads in doubles.

The reference decode chunk is now bounded by its score scratch: four tokens with 32 heads and 170K keys exceeded
the test's 64 MiB buffer. That first test attempt failed due to reference buffer corruption; the passing logs
are after bounding the chunk. The group-8 shape still uses chunk 4.

Kernel timings are CUDA-event samples from the correctness suite, **not repeated whole-model benchmarks**.
Prompt timings have one sample; decode timings average ten launches after a warm-up.

| Group 16 case | Optional kernel, ms | Existing path, ms |
| --- | ---: | ---: |
| Q8, 30K position, prompt batch 1024 | 17.56 | 45.71 |
| FP16, 120K position, prompt batch 1024 | 67.47 | 173.08 |
| Q8, 140K position, one decode token | 0.420 | 2.571 |
| FP16, 140K position, one decode token | 0.776 | 1.650 |
| Q8, 1024 position, one decode token | 0.028 | 0.018 |

Short-context decode can be slower. These numbers measure one attention call; CPU experts and other layers
can dominate whole-model latency. Full logs are alongside this report.

The unchanged HIP stubs also compile with:

```sh
g++ -std=c++17 -x c++ -DGGML_USE_HIP -Iinclude -c src/poly/fattn.cu -o fattn-hip-stubs.o
```

This is a host compilation of the excluded CUDA path, **not a full ROCm backend build or AMD runtime test**.
No SYCL source or target changes. No results on other NVIDIA architectures, Windows, multi-GPU or vision.

## Real-model check

Model: `AtomicChat/Ornith-1.5-122B-Uncensored-GGUF`,
`Ornith-1.5-122B-Uncensored-AD-IQ4_XS_S/Ornith-1.5-122B-Uncensored-AD-IQ4_XS_S-00001-of-00038.gguf`
and the other 37 shards. Model shard sizes and LFS SHA256 values are recorded in `model-files.json`.
No vision encoder is loaded. The tokenizer was extracted from the first shard with `tools/strata_tokenizer.py`.

`check_model.py` supplies exact prompts, arguments and alternating order for three opt-out/opt-in rounds. It
starts a fresh engine for every run, with no saved expert profile, frozen expert placement, expert swaps off,
Q8 KV, 32,768 maximum context with KV allocated at load, 12 CPU threads, 1024 MiB reserve, greedy sampling,
lookup off and tail skip off. CUDA graphs remain on. The default-output check disables MTP; the timed runs
keep the default MTP window. Loading is excluded from the engine's throughput timings; OS page cache is retained.
These deliberately controlled settings differ from a production service with an adapted expert profile.

```sh
python bench/results/2026-10-21-ornith-gqa16/check_model.py \
  --baseline RELEASE_ENGINE --engine build/strata-poly --model FIRST_SHARD \
  --tokenizer PACK/tokenizer --out RESULTS
```

The official release and new build with both opt-ins unset produced the same 87 generated token IDs on the
24-token coding prompt (SHA256 `cdc486079afafebdb8242694571236aac5102629da8ee313d6791e77c8e015d3`).
This is a default-path regression check on one prompt, not proof of all possible outputs. Their timings are
not used for a speed claim because page-cache warm-up differed.

Whole-model results are recorded in `model-8k/results.json` and `model-16k/results.json`; inputs and token
outputs are included for reproduction. Engine logs were saved with a `.txt` extension for version control.

| Fresh prompt | Opt-ins off, prompt tok/s | Both on, prompt tok/s | Runs per arm |
| --- | --- | --- | ---: |
| 8,440 tokens | median 797.2 (729.7–798.9) | median 779.0 (719.0–832.2) | 3 |
| 16,840 tokens | 927.2 | 1013.8 | 1 |

The 8K ranges overlap: no reliable end-to-end improvement established there. The single 16K pair is about
9% faster reading with both opt-ins, but is not a repeated estimate. All eight long-prompt runs returned the
same five token IDs, the correct `7391` and end marker. Timed runs all loaded 1204 / 12544 experts (5.8 GiB)
into VRAM. To repeat the 16K pair add `--repeats 1400 --rounds 1 --skip-default` and a separate output folder.
An attempted 32K CLI input hit Linux's per-argument size limit before the engine started; larger inputs should
use the engine's stdin server protocol rather than this command-line harness.

The long prompt asks for a planted numeric code, so generated answers are short. Decode throughput on this
workload must not be generalized to sustained generation. No end-to-end 256K run or overall model-quality
evaluation was performed.

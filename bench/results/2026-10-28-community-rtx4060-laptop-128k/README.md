# RTX 4060 Laptop 8 GB, Qwen3.6-35B-A3B at 118K: the two opt-in attention kernels, and what the prompt cache gives back

Measured on 2026-10-28 on one laptop, PolyStrata 0.1.45 (engine 0.1.45). The question was narrow: at a 128K
context — the size an agentic coding session needs — what is left in prompt reading and in writing, and does
the prompt cache give back what it should? Everything below is one machine, one model file, eight arms, no
quality evaluation.

This is a report, not a change: nothing here alters a default.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GiB VRAM (every reading across the eight arms was 7,299-7,795 MiB;
  the two ends are the baseline arm itself, see below)
- CPU: Intel Core i9-14900HX, 8 performance + 16 efficiency cores, 8 CPU threads for the experts that are not in VRAM
- RAM: 31.7 GiB installed; storage: not measured; mains power
- OS: Windows 11 (10.0.26200); GPU driver 621.01; CUDA: not read
- PolyStrata 0.1.45, commit `145493e`; engine 0.1.45, ready-made `strata-poly.exe`
- No other GPU or model workload ran; background load and power limits were not recorded

## Model and configuration

- `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`, Unsloth UD-Q3_K_M, the size setup installs for 32 GB of RAM
- 2,470-2,473 of 10,496 routed experts in VRAM (3.2 GiB) at every start, expert profile warm
- Context 131,072; `--kv q8_0`; `--tail-skip 6`; `--vram-reserve-mib 512`; `--threads 8`;
  `--conversation-cache-mib auto` (3,072-4,096 MiB, 4 slots, sized from free RAM at each start)

The baseline arm is the installed configuration unchanged. The other arms change only what the table names.

## Method

Three request shapes, because the cache figures depend on the client and the timings do not:

1. **Kernel arms (A-D), and the checkpoint arms (E-F).** One user message of 117,977 tokens: the
   concatenated text of `docs/DETAILS.md`, `docs/BACKENDS.md` and `docs/AMD_HIP.md`, trimmed to 360,000
   characters. Turn 2 appends a ~50-token question **to the same user message**, which is the worst case for
   the cache: the prefix diverges right after the prompt.
2. **Arm G, chat shape.** `[system, user]`, then `[system, user, assistant: the model's own answer, user]`,
   which is what a client that resends the conversation sends. Plus a third turn that repeats turn 2's
   messages byte for byte, as a client retry.
3. **Arm H, harness shape.** The same as G plus 12 tool schemas in the `tools` field and a tool result
   appended to the follow-up, which is the shape DeepSeek Harness (and other agent harnesses) sends to a
   local model. Prompt trimmed to 330,000 characters so system + tools + prompt stay inside 128K.

Every arm: its own server process and its own config (`raw/cfg-*.json`), one request per turn, answer cap 64
tokens, temperature 0, non-streaming. **Arms A-D were run twice** (two rounds, about an hour apart, each in a
fresh process) so the kernel table has ranges; E-H were run once. Prompt and decode speeds are the engine's
own timings (`prompt_per_second`, `predicted_per_second`); `cached_tokens` is the server's
`prompt_tokens_details`.

## Results

### The two kernels, 117,977-token prompt — two runs per arm

Arm A-D were run twice, at 20:0x and again at 21:0x, each arm in a fresh process with its own config.
Medians over the two runs, with the range:

| Arm | Env / args | Prompt tok/s | Decode tok/s | Turn 1 | Turn 2 read | Turn 2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A baseline | as installed | 814 (806-822) | 32.1 (32.0-32.1) | 145.3-148.0 s | 11,519 | 25.1-25.2 s |
| B | `STRATA_FATTN=1` | **1,318** (1,312-1,323) | 34.2 (33.5-34.9) | 90.7-91.5 s | 11,519 | 14.7-14.8 s |
| C | `STRATA_FDEC=1` | 830 (829-830) | **51.0** (50.1-51.9) | 143.2-143.4 s | 11,519 | 23.9-24.1 s |
| D | B + C + `--stage-read-mib 512` | **1,308** (1,303-1,312) | **51.4** (50.7-52.0) | 91.0-93.8 s | 11,519 | 14.3-14.4 s |

Against A, the median ratios are **1.62x prompt** and **1.60x decode**. Round to round, the 117,977-token
prefill moved by at most 1.9% (A itself, 806.2 against 821.7) and turn-1 decode by at most 4.1% (B, 34.9
against 33.5); the two ratios are far outside both, but a decode difference of a few percent at two runs is
not resolvable here. C's turn-2 decode is the one cell that moved a lot between rounds (50.7 against 62.1,
20%), which is why the first turn is the number used for the comparison.

VRAM does not track the arm: every reading in the eight runs sat between 7,299 and 7,795 MiB, and the lowest
and the highest are both the baseline arm's own (`raw/` has the per-request numbers, idle and after each
turn). The spread is the desktop and background programs, not an allocation: neither switch adds one.

The two kernels are independent: B's decode (34.2, against A's 32.1) and C's prompt (830, against A's 814)
are close to the noise. `--stage-read-mib 512` neither helped nor hurt at this context (B and D are level).

One detail worth recording: on the follow-up turn, decode in arms C and D is **faster** than on the first
turn (62.1 and 60.6 tok/s against 50.1 and 50.7), because that turn re-reads only 11,519 tokens and the
expert cache has been adapted by the first one. The first turn's decode is the number to compare with A.

### The prompt cache: checkpoints, not parking, decide a follow-up turn

| Arm | Change from D | Turn 2 read | Reused | Turn 2 |
| --- | --- | ---: | ---: | ---: |
| D | — | 11,519 | 106,496 | 14.4 s |
| E | `--prompt-cache 16 --prompt-cache-every 1024` | **5,375** | 112,640 | **8.1 s** |
| F | `--prompt-cache 0` | **118,015** | 0 | 91.3 s |

F's turn 1 is a cold-start outlier (664 tok/s) and is not used: its turn 2 read the same 118K tokens at
1,306 tok/s, so the machine was not slow. F matters only for its turn-2 row: **with checkpoints off, a
modified prompt cannot resume at all**, so parking alone does not cover this case — the parked conversation
ends after the generated tokens, while the modified prompt diverges before that.

### What a real client gets

| Turn | Prompt read | Reused | Wall |
| --- | ---: | ---: | ---: |
| G1 chat shape: fresh 117,995-token prompt | 117,995 | — | 89.8 s (1,330 tok/s) |
| G2 follow-up with the assistant's reply resent | 5,395 | 112,640 | 7.8 s |
| G3 the same messages again (client retry) | **1** | 118,034 | **1.0 s** |
| H1 harness shape: system + 12 tools + 108K prompt (111,318 tokens) | 111,318 | — | 82.4 s (1,373 tok/s) |
| H2 after a tool result (111,425 tokens) | 8,001 | 103,424 (92.8%) | 10.4 s |

So at ~110K tokens with an agent prefix: **82 s for the first turn, ~10 s per follow-up**, and 1.0 s when the
client resends identical messages. The 5,395-8,001 tokens a follow-up reads again are the gap to the newest
usable checkpoint plus the chat template around the assistant and tool messages, not a cache miss.

### A client can defeat the cache, and that is worth knowing

If a harness edits or replaces a message *earlier* in the history — a runtime-context block that "supersedes"
the previous one, a re-rendered system prompt, a timestamp at the top, a compacted history — the prefix no
longer matches and the engine reads the whole prompt again. The tell is `cached_tokens` per answer (or
`/metrics` → `conversation_cache`): a stable conversation whose second turn reports ~0 reused is being
rewritten by the client, not mis-cached by the engine. Nothing in this report tests that case; it is listed
because it is the first thing to check before blaming the cache.

## Correctness and limitations

- **No quality test.** Not measured for any arm: this is a speed report. The project's own pages document
  what the two kernels cost numerically: the flash prompt kernel's products are up to 7.7e-3 from the
  arithmetic in doubles where the default path is within 2.3e-3, greedy answers part from the default path's
  after a few dozen characters at near-ties, 11 of 12 checkable long-context answers were unchanged, and
  recall passed at 107K and 247K; the tensor-core decode kernel is within 1.8e-3, with the same 11/12 and the
  same recall results.
- **Two runs for the kernel arms (A-D), one for the rest (E-H).** The kernel ratios held to within 1.5%
  between the two rounds. The cache arms were not repeated; their token counts are exact (11,519 / 5,375 /
  118,015) and did not vary between rounds in A-D, but their wall times are single runs.
- **The cache rows are client-shaped.** Arms A-F use an artificial prompt (a question appended to the same
  user message); G and H are closer to a real client. A harness that rewrites history is not covered at all.
- **Not tested:** streaming (every request was non-streaming), concurrent requests, thinking on, pictures,
  other quantization sizes, other cards, and `--prompt-batch` or `--skip-weight` at this context.
- Machine details came from the OS and `nvidia-smi`; background load and power limits were not recorded.

## Reproduce

The prompt is the concatenation described above; `scripts/` holds the four scripts as they ran, with local
paths and the API key replaced by placeholders (`<STRATA>`, `<DATA>`, `<API_KEY>`). Put a config of your own
in place of `raw/cfg-*.json` and run from the PolyStrata folder:

```powershell
python bench\results\2026-10-28-community-rtx4060-laptop-128k\scripts\run_arms.py    # A-D
$env:PYTHONPATH = 'bench\results\2026-10-28-community-rtx4060-laptop-128k\scripts'
python bench\results\2026-10-28-community-rtx4060-laptop-128k\scripts\run_arms2.py   # E-F
python bench\results\2026-10-28-community-rtx4060-laptop-128k\scripts\run_arms3.py   # G
python bench\results\2026-10-28-community-rtx4060-laptop-128k\scripts\run_arms4.py   # H
```

`raw/results-AD-round1.json` and `raw/results-AD-round2.json` hold the two rounds of the kernel arms,
`raw/results-EF.json` the checkpoint arms, `raw/results-G.json` and `raw/results-H.json` the two
harness-shaped ones: every request, with prompt and answer token counts, timings, cache figures and VRAM.
`raw/engine-logs/` has each arm's own engine lines, which say how many experts were cached, which kernel
was active and what the conversation cache did.

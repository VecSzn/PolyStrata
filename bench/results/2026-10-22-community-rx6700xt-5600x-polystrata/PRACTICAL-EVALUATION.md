# Practical evaluation, separately labeled (not the speed suite)

This file is **not** part of the official benchmark suite and is not
comparable to it. It records what this machine's campaign measured beyond
speed, for readers deciding between models on 12 GB RDNA2 hardware.
Methodology and caveats per section; no general accuracy claim is made.

## A. Reliability: 6-hour day-desktop soak (2026-10-21)

The ctx32768 Qwen server (the daily driver) served **483 health checks
and traffic over 6 h under normal desktop use** (browser, editor) with
**zero failures**, GPU telemetry live (`backend: hip`). Summary:
`soak-summary.json` next to this file. The summary's `max_server_rss_kB`
tracks the Python wrapper process, not the engine (engine RSS is tens of
GB, mostly reclaimable page cache) — ignore that field. This is a single
soak, not a statistical claim.

## B. Blind agent screens (12-task easy set, 2026-10-20)

An agent client (Nanobot, thinking on, temp 0.6, 20-tool-call cap)
completed a 12-task office/research/synthesis suite twice, once per
model, blind: rows and workspaces merged under symlinked names, graded
against fixture-derived ground truths, grade reasons sealed by SHA before
unblinding.

| | Total | coding | research | synthesis | drafting |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6 Q4_K_XL (poly) | **10/12** | 3/3 | 2/3 | **2/3** | 3/3 |
| LFM2.5-24B Q4_K_XL (poly) | 8/12 | 3/3 | 2/3 | 0/3 | 3/3 |

Both failed the same two research tasks (a procurement-revision conflict
and a count-preservation check — the failure class: a fact wired to the
wrong slot). LFM's synthesis failures included invented measurement
values. Caveats: 1 repetition per task; same fixtures for both arms;
graders saw neither model identity nor arm.

## C. Hard-set screen (6 discriminating tasks, 2026-10-21/22)

Six tasks built to discriminate (generator-derived ground truths: fixed
totals, ID sets, a windowed timeline, a revision conflict, a sandboxed
code task), screened blind the same way. The easy set's models all
saturated it except Gemma:

| Model | hard set | probe | gate |
| --- | --- | --- | --- |
| Qwen3.6 Q4_K_XL (driver) | **6/6** | 4/4 | 12/12 |
| Tiel Coder Q3_K_XL | **6/6** | 4/4 | 11/12 |
| Gemma-4 QAT (speed arm) | **3/6** | 4/4 | 10/12 |

Gemma's failures: an arithmetic total (211 vs 216), an unwritten answer
file, the sandbox code task, and a recovery sequence (twice). Caveats:
1 repetition, 6 tasks, one failure was a degenerate generation; four
independent signals agree on direction.

## D. Same-day llama.cpp controls (methodology check)

The matched harness (identical prompts across engines) ran same-day
llama.cpp controls on this box: the current llama.cpp default
(IQ3_XXS arm) measured 60.2/66.1 tok/s synth/prose against
strata-poly's Qwen 75.4/73.5 on the same prompts; a fresh control the
next day drifted ~2%, bounding day-to-day noise. These are custom
matched probes, not suite numbers, and are reported only to show the
speed ordering is not a harness artifact.

## What was not tested

Three repetitions (screens are 1x); other agent clients; other
quantizations of the same models on this card; the draft/spec state
separately (engine default); `--tail-skip 6`; hipBLASLt tuning for
gfx1031; concurrency beyond the suite's 2-client stage.

# Custom paired Qwen tail-skip comparison

This is a custom experiment, not an official `tools/strata_bench.py` suite; it
reuses upstream HTTP helpers. It tests the original
Qwen3.6 UD-Q3_K_M MTP file on the same executable, configured context65536,
FP16 KV, `--kv-whole`,16 CPU workers and automatic conversation cache.
Only `STRATA_ROUTE_TAIL_SKIP` changes between6 and0. The identical CLI retains
`--tail-skip 6`; upstream `src/poly/main.cpp` reads the environment afterward,
so environment0 overrides it. Running engine environments were observed during
the experiment, not inferred only from launch arguments.

Three matched pairs ran in AB / BA / AB order, A=6 and B=0. Each arm started a
fresh server with an exact copy of [initial-profile.txt](tail-skip/initial-profile.txt),
SHA256 `1cfe73631867a7cf28fabb5fbfa4079c7793b14129316d8d1f6ede0e7e550a07`.
There were six servers and30 requests including warm-ups, with zero request errors.
The immutable model/engine/profile identities are in [provenance.json](tail-skip/provenance.json).

Each arm ran the same block: warm-up → strict sort → coding →384-token generation
→ long retrieval. The starting profile was reset per arm, **not per task**.
All requests used temperature0 and thinking off; no API sampling seed was set.
The profile and adaptive cache can evolve differently after different answers.
Long requests had zero reused prompt tokens; this does not establish cold model,
expert or filesystem caches. GPU budgets ranged6369–6374MiB during these starts.

## Results

Means ± sample standard deviation, n=3 per arm; all per-arm responses, grades,
timings, token counts, GPU budgets and memory aggregates are in
[results.json](tail-skip/results.json). No statistical-significance claim is made.

| Metric | Tail skip6 | Tail skip0 |
|---|---:|---:|
|384-generated-token task, decode tok/s |31.23 ±0.31 |29.77 ±0.85 |
|Long prompt processing tok/s |345.70 ±0.36 |345.73 ±0.21 |
|Long wall TTFT seconds |168.162 |168.162 |
|Long48-token reply, decode tok/s |25.50 ±0.26 |23.27 ±1.72 |
|Long wall total seconds |170.041 |170.232 |
|Coding decode tok/s |33.60 ±0.66 |31.37 ±0.76 |

Skipping disabled had approximately4.7% lower throughput on the384-token task.
The paired generated texts differ, so this is not identical-token replay; the
generation cap can truncate answers. Coding output lengths/content also differ.
The48-token retrieval reply is a short decode sample. Prefill was essentially
unchanged in this experiment; that does not establish a universal speed effect.

| Bounded correctness check | Tail skip6 | Tail skip0 |
|---|---:|---:|
|Strict JSON sort |2/3 |0/3 (Markdown fences) |
|Coding, all8 functional cases and strict format |2/3 |3/3 |
|Long strict JSON and all3 facts |3/3 |3/3 |

One tail6 coding answer merged one-unit gaps and passed6/8 cases; the other two
passed8/8. The three tail0 coding answers passed8/8. These are one coding task
and one sorting task, repeated with fresh servers; quality is mixed, with no
general accuracy winner. All six retrievals exercised58117 input tokens and
generated48 tokens. Their output cap was384, distinct from the earlier128-token
single-probe allowance.

Across arms, sampled peaks reached35.890GiB whole-system RAM,17.811GiB engine
RSS and2.345GiB engine private commit. Categories overlap; these are not additive.
Per-arm baselines and dedicated/shared GPU counters are retained. The Windows
per-instance activity counters occasionally reported values above100; the
retained raw maximum is not a normalized whole-GPU utilization percentage.

## Reproduce the block

[tasks.json](tail-skip/tasks.json) retains exact non-long request bodies and a
fixed generator/hash for the long prompt. The original Windows file used CRLF;
the API string used LF. Both identities are explicitly distinguished in provenance.
The model, executable, FP16/kv-whole configuration and initial profile must match.

1. Copy the supplied profile to a new writable path for each arm. Point a copy
   of the original Qwen config at it, set max-context65536, and keep the other
   settings unchanged. Do not reuse a modified profile from an earlier arm.
2. In the server's launching shell, set `$env:STRATA_ROUTE_TAIL_SKIP='6'` or `'0'`
   as required by the AB/BA/AB order, then start one fresh server. Confirm the
   running engine inherited the selected value; do not run two arms concurrently.
3. Send the block below once. Stop the server afterward, preserve its changed
   profile/logs, and repeat with a new copy of the frozen initial profile.

```powershell
python reproduce-tail.py --verify-only
python reproduce-tail.py --source <checkout> --initial-profile <fresh-profile-copy>
```

The helper verifies the model/context and initial profile hash, sends the same
ordered tasks, and prints raw client responses and metrics. It does not launch
servers, set their environment or reset profiles between tasks. The original
restricted coding grader and eight cases are provided at the report root.

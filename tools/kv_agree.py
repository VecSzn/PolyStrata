#!/usr/bin/env python3
"""Teacher-forced agreement of a KV cache format against fp16's greedy picks.

    python tools/kv_agree.py --model M.gguf [--ref fp16] [--kv q8_0]

For each of --prompts random prompts (ids in --lo..--hi, a fixed seed), the reference arm writes
--max-new tokens greedily.  The other arm is then run with --follow of those tokens (--window 1,
no draft, no prompt or conversation cache, no EOS stop) and --top-probs 1, and its most likely
token at every step is compared with the reference's pick.  One line a prompt, then the total;
with --min-agree the exit status is 1 below it.

Options after -- extra go to both engine runs, e.g. -- --tail-skip 0.
"""
import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(binary, model, kv, ids, a, follow=None):
    cmd = [binary, "--model", model, "--kv", kv, "--tokens", ",".join(map(str, ids)),
           "--max-new", str(a.max_new), "--max-context", str(a.max_context), "--window", "1", "--lookup", "0",
           "--no-draft", "--prompt-cache", "0", "--conversation-cache-mib", "0", "--eos-ids", "0"]
    if follow is not None:
        cmd += ["--follow", ",".join(map(str, follow)), "--top-probs", "1"]
    cmd += a.passthrough
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit(f"exit {p.returncode} for --kv {kv}")
    return p.stdout, p.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", default=str(ROOT / "build" / "strata-poly"))
    ap.add_argument("--ref", default="fp16", help="the arm whose greedy picks are the reference (default fp16)")
    ap.add_argument("--kv", default="q8_0", help="the arm under test (default q8_0)")
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--length", type=int, default=32, help="prompt ids, the first 1 (BOS), the rest random")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--lo", type=int, default=1000, help="random prompt ids are drawn from --lo..--hi")
    ap.add_argument("--hi", type=int, default=60000)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--min-agree", type=float, default=0.0, help="exit 1 below this total agreement")
    ap.add_argument("passthrough", nargs="*", help="extra engine options, after --")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    total = same = 0
    for i in range(a.prompts):
        prompt = [1] + [rng.randrange(a.lo, a.hi) for _ in range(max(0, a.length - 1))]
        out, _ = run(a.bin, a.model, a.ref, prompt, a)
        got = [int(x) for x in re.findall(r"-?\d+", out)][:a.max_new]
        if not got:
            raise SystemExit(f"prompt {i + 1}: the {a.ref} arm wrote no token ids")
        _, err = run(a.bin, a.model, a.kv, prompt, a, follow=got)
        tops = [int(m.group(1)) for m in re.finditer(r"^top (\d+):", err, re.M)]
        tops = tops[:len(got)]
        n = min(len(got), len(tops))
        hit = sum(1 for k in range(n) if got[k] == tops[k])
        total += n
        same += hit
        print(f"prompt {i + 1}: {hit}/{n} of {a.ref}'s picks are {a.kv}'s most likely "
              f"({100.0 * hit / max(n, 1):.1f}%)")
        sys.stdout.flush()
    agree = same / max(total, 1)
    print(f"total: {same}/{total} = {100.0 * agree:.2f}% ({a.kv} against {a.ref})")
    return 0 if agree >= a.min_agree else 1


if __name__ == "__main__":
    sys.exit(main())

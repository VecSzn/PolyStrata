"""Offline, fresh-process Q8 checks; no server, prefix reuse, or saved expert profile.

python check_model.py --baseline OLD --engine NEW --model FIRST.gguf \
    --tokenizer PACK/tokenizer --out RESULTS
Run with only the selected test GPU visible. Loading is outside the engine's timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from strata_tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    for name in ("baseline", "engine", "model", "tokenizer", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--repeats", type=int, default=700, help="filler repetitions (700: 8440 tokens; 1400: 16840)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--skip-default", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_dir(a.tokenizer)
    short = "Write a Python function that returns the square of an integer."
    filler = "river stone market lantern copper winter garden signal harbor needle orbit velvet " * a.repeats
    long = "Remember: the archive access code is 7391.\n" + filler + "\nWhat is the archive access code? Reply with just the number."

    def ids(text):
        return tok.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)

    inputs = {"short": ids(short), "long": ids(long)}
    if len(",".join(map(str, inputs["long"]))) > 120000:
        ap.error("token argument exceeds this CLI harness's limit; use fewer repetitions or the server protocol")
    (out / "inputs.json").write_text(json.dumps(inputs))
    rows = []

    def run(tag, exe, prompt, enabled=False, draft=True):
        env = dict(os.environ, STRATA_FREEZE_EXPERTS="1")
        for key in ("STRATA_FATTN", "STRATA_FDEC"):
            env.pop(key, None)
            if enabled:
                env[key] = "1"
        cmd = [exe, "--model", a.model, "--tokens", ",".join(map(str, inputs[prompt])),
               "--max-new", "96", "--max-context", "32768", "--threads", "12",
               "--vram-reserve-mib", "1024", "--kv", "q8_0",
               "--kv-whole", "--expert-swaps", "0", "--lookup", "0", "--tail-skip", "0",
               "--temperature", "0", "--seed", "7"]
        if not draft:
            cmd += ["--no-draft", "--window", "1"]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        # Engine logs contain the local model path: replace it before sharing.
        log = p.stderr.replace(a.model, "<MODEL>")
        (out / (tag + ".log")).write_text(log)
        (out / (tag + ".ids")).write_text(p.stdout)
        match = re.search(r"prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s", log)
        row = {"tag": tag, "exit": p.returncode, "enabled": enabled, "draft": draft,
               "input_tokens": len(inputs[prompt]), "output_sha256": hashlib.sha256(p.stdout.encode()).hexdigest()}
        if match:
            row.update(prompt_n=int(match[1]), prompt_tps=float(match[2]), generated_n=int(match[3]), decode_tps=float(match[4]))
        if p.returncode == 0:
            row["text"] = tok.decode([int(x) for x in p.stdout.strip().split(",") if x])
        rows.append(row)
        (out / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
        if p.returncode or not match:
            raise RuntimeError("failed: " + tag + "; see log")
        return p.stdout

    if not a.skip_default:
        old = run("release-default", a.baseline, "short", draft=False)
        new = run("new-default", a.engine, "short", draft=False)
        if old != new:
            raise AssertionError("default token streams differ")
    for round in range(a.rounds):
        for enabled in ([False, True] if round % 2 == 0 else [True, False]):
            run(f"long-r{round}-{'on' if enabled else 'off'}", a.engine, "long", enabled=enabled)


if __name__ == "__main__":
    main()

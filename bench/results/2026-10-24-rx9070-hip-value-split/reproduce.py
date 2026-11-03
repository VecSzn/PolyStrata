#!/usr/bin/env python3
"""Measure an already-running, independently restarted strata-poly server.

Restart with the documented engine settings and a fresh copy of the same expert
profile before each arm. This script never changes models or server settings.
The API key is read from an environment variable, never written to the report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
BASE = "c174f35c2a5e20a4fe7ac236063d6513fcc07179"
TASK = "Write a complete Python LRU cache implementation with type hints, an explanation of its invariants, and six unittest test cases. Include the full code and detailed explanation."

def corpus(root):
    if root:
        return "\n".join(p.read_text(errors="replace") for p in sorted((root / "docs").glob("*.md")))
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE, "docs"], cwd=ROOT, text=True)
    names = sorted(n for n in names.splitlines() if n.count("/") == 1 and n.endswith(".md"))
    if not names:
        raise RuntimeError("base commit unavailable; fetch it or use --corpus-root with its source archive")
    texts = [subprocess.check_output(["git", "show", BASE + ":" + n], cwd=ROOT).decode("utf-8", "replace")
             .replace("\r\n", "\n").replace("\r", "\n") for n in names]
    return "\n".join(texts)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18090")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--arm", required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--mtp", action="store_true", help="label an MTP-on engine; use a 768-token output cap")
    p.add_argument("--depths", type=int, nargs="+", default=[1024, 32768, 98304])
    p.add_argument("--api-key-env", default="POLYSTRATA_API_KEY")
    p.add_argument("--out", type=Path, default=Path("results.jsonl"))
    p.add_argument("--corpus-root", type=Path, help="pristine base archive, if the git base is unavailable")
    p.add_argument("--dry-run", action="store_true", help="write prompt hashes without contacting a server")
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / "tools"))
    from strata_tokenizer import Tokenizer
    tok = Tokenizer.from_dir(args.tokenizer)
    text = corpus(args.corpus_root)
    ids = tok.encode(text * 3)
    if min(args.depths) < 0 or max(args.depths) > len(ids):
        raise ValueError("depth outside the available corpus")
    headers = {"Content-Type": "application/json"}
    if os.environ.get(args.api_key_env):
        headers["Authorization"] = "Bearer " + os.environ[args.api_key_env]

    def request(path, body=None):
        req = urllib.request.Request(args.url.rstrip("/") + path, headers=headers,
                                     data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=240) as r:
            return json.load(r)

    def ask(prompt, cap):
        return request("/v1/chat/completions", {
            "model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": cap,
            "temperature": 0, "top_p": 1, "top_k": 1, "seed": 42, "repetition_penalty": 1,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})

    if not args.dry_run:
        health = request("/health")
        if not health.get("loaded") or health.get("model") != args.model:
            raise RuntimeError("server is not loaded with the requested model")
        ask("Warm-up: write a Python function that adds two integers.", 32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for depth in args.depths:
        prompt = f"Independent benchmark sample {args.rep}, context {depth}.\n" + tok.decode(ids[:depth])
        prompt += "\nIgnore the reference material above for this task.\n" + TASK
        row = {"arm": args.arm, "rep": args.rep, "depth": depth, "mtp": args.mtp,
               "corpus_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        if not args.dry_run:
            start = time.monotonic()
            answer = ask(prompt, 768 if args.mtp else 512)
            row.update(elapsed_s=time.monotonic() - start, timings=answer.get("timings"),
                       usage=answer.get("usage"), finish=answer["choices"][0].get("finish_reason"),
                       message=answer["choices"][0].get("message"))
        with args.out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "message"}))

if __name__ == "__main__":
    main()

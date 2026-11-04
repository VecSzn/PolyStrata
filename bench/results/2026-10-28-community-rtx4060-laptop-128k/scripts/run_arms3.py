"""Harness-shaped test: does the parked conversation restore when turn 2 is a
real chat history (user, assistant, user) instead of a modified user message?

Arms A-D appended a question to the SAME user message, so the prefix diverged
right after the prompt and the engine had to rewind to a checkpoint. A real
agent harness resends the assistant's own reply, which should match the parked
state exactly.

One arm, the winning configuration (FATTN + FDEC + 16 checkpoints):
  turn 1  [system, user: 118K prompt]                  -> 64 tokens
  turn 2  [system, user, assistant: that answer, user] -> 64 tokens
Reports prompt_n / cached tokens for both turns, plus a third turn that is a
pure repeat of turn 2's messages (a client retry: must be a full cache hit).
"""
import copy
import json
import os
import subprocess
import time

import requests

ROOT = r"<STRATA>"
PY = os.path.join(ROOT, r".venv\Scripts\python.exe")
OUT = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(OUT, "prompt.txt"), encoding="utf-8").read()
PORT = 8097
URL = f"http://127.0.0.1:{PORT}"
CFG_SRC = os.path.join(ROOT, "strata-qwen36-ud-q3_k_m-128k-fast.json")

SYSTEM = "You are a coding agent working in a repository. Answer briefly."
NEXT_TURN = "Now list the three files this engine reads at start-up, one line each."


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ask(key, messages, max_tokens=64):
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": "qwen3.6-35b-a3b", "messages": messages,
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    t0 = time.time()
    r = requests.post(URL + "/v1/chat/completions", headers=h, json=body, timeout=1800)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall


def main():
    cfg = json.load(open(CFG_SRC, encoding="utf-8"))
    cfg["port"] = PORT
    cfg["log"] = os.path.join(OUT, "engine-G-harness.log")
    cfg_path = os.path.join(OUT, "cfg-G-harness.json")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=1)
    key = cfg["api_key"]

    srv = open(os.path.join(OUT, "server-G-harness.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([PY, os.path.join(ROOT, "serve", "server.py"), "--engine", "strata",
                             "--config", cfg_path, "--port", str(PORT)],
                            cwd=ROOT, stdout=srv, stderr=subprocess.STDOUT)
    out = {"arm": "G-harness-chat", "turns": []}
    try:
        h = {"Authorization": f"Bearer {key}"}
        for _ in range(300):
            try:
                if requests.get(URL + "/v1/models", headers=h, timeout=10).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(3)

        base = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT}]
        resp, wall = ask(key, base)
        answer = resp["choices"][0]["message"]["content"] or ""
        tm = resp.get("timings") or {}
        out["turns"].append({"turn": "1 fresh prompt", "wall_s": round(wall, 1),
                             "prompt_n": tm.get("prompt_n"), "prompt_tok_s": tm.get("prompt_per_second"),
                             "decode_tok_s": tm.get("predicted_per_second"),
                             "cached": None, "answer_chars": len(answer)})
        log(f"turn1 prompt {tm.get('prompt_n')} tok @ {tm.get('prompt_per_second'):.0f} tok/s, "
            f"answer {len(answer)} chars, wall {wall:.1f}s")

        chat = base + [{"role": "assistant", "content": answer},
                       {"role": "user", "content": NEXT_TURN}]
        for name, msgs in (("2 follow-up turn (assistant resent)", chat),
                           ("3 identical resend (retry)", chat)):
            resp, wall = ask(key, msgs)
            tm = resp.get("timings") or {}
            usage = resp.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            out["turns"].append({"turn": name, "wall_s": round(wall, 1),
                                 "prompt_n": tm.get("prompt_n"),
                                 "prompt_tok_s": tm.get("prompt_per_second"),
                                 "decode_tok_s": tm.get("predicted_per_second"),
                                 "cached": cached})
            log(f"{name}: read {tm.get('prompt_n')} tok, cached {cached}, wall {wall:.1f}s")
            json.dump(out, open(os.path.join(OUT, "results-G.json"), "w", encoding="utf-8"), indent=1)

        log("--- engine log ---")
        for ln in open(cfg["log"], encoding="utf-8", errors="replace"):
            if "conversation cache" in ln or "restored" in ln or "parked" in ln:
                log(ln.strip())
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, text=True)
        srv.close()
    json.dump(out, open(os.path.join(OUT, "results-G.json"), "w", encoding="utf-8"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()

"""A/B of strata-poly's opt-in attention paths at ~100K context, on one machine.

Arms:
  A baseline            (config as installed)
  B STRATA_FATTN=1      (flash-attention prompt kernel)
  C STRATA_FDEC=1       (tensor-core decode attention)
  D both + --stage-read-mib 512

Each arm: start the server with its own config, wait until ready, send the
same ~100K-token prompt (turn 1), then the same prompt plus a short suffix
(turn 2, to see how much of the prefix the cache reuses), then stop it.

Results are appended to results.json after every request, so a crash keeps
what was measured.
"""
import copy
import json
import os
import subprocess
import sys
import time

import requests

ROOT = r"<STRATA>"
PY = os.path.join(ROOT, r".venv\Scripts\python.exe")
OUT = os.path.dirname(os.path.abspath(__file__))
BASE_CFG = os.path.join(ROOT, "strata-qwen36-ud-q3_k_m-128k.json")
PROMPT = open(os.path.join(OUT, "prompt.txt"), encoding="utf-8").read()
PORT = 8099
URL = f"http://127.0.0.1:{PORT}"

SUFFIX = (
    "\n\n---\n\nOne more thing, now that you have read all of the above: "
    "list the three files this engine reads at start-up and say in one sentence "
    "what each one is for."
)

ARMS = [
    ("A-baseline", {}, []),
    ("B-fattn", {"STRATA_FATTN": "1"}, []),
    ("C-fdec", {"STRATA_FDEC": "1"}, []),
    ("D-both", {"STRATA_FATTN": "1", "STRATA_FDEC": "1"}, ["--stage-read-mib", "512"]),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vram_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def kill_tree(proc):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                   capture_output=True, text=True)


def wait_ready(key, timeout=900):
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(URL + "/v1/models", headers=headers, timeout=10)
            if r.status_code == 200:
                return time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError("server did not become ready")


def ask(key, content, max_tokens=64):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(URL + "/v1/chat/completions", headers=headers, json=body, timeout=1800)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall


def engine_lines(log_path):
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return [ln.rstrip() for ln in f if "strata-poly:" in ln or "[strata]" in ln][-14:]
    except OSError:
        return []


def main():
    base = json.load(open(BASE_CFG, encoding="utf-8"))
    key = base["api_key"]
    results = []
    out_json = os.path.join(OUT, "results.json")

    for name, env, extra in ARMS:
        cfg = copy.deepcopy(base)
        cfg["port"] = PORT
        cfg["args"] = list(cfg["args"]) + extra
        cfg["env"] = {**(cfg.get("env") or {}), **env}
        cfg["log"] = os.path.join(OUT, f"engine-{name}.log")
        cfg_path = os.path.join(OUT, f"cfg-{name}.json")
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=1)

        arm = {"arm": name, "env": env, "extra_args": extra, "turns": []}
        log(f"=== {name} env={env} extra={extra}")
        srv_log = open(os.path.join(OUT, f"server-{name}.log"), "w", encoding="utf-8")
        proc = subprocess.Popen(
            [PY, os.path.join(ROOT, "serve", "server.py"), "--engine", "strata",
             "--config", cfg_path, "--port", str(PORT)],
            cwd=ROOT, stdout=srv_log, stderr=subprocess.STDOUT)
        try:
            ready = wait_ready(key)
            arm["ready_s"] = round(ready, 1)
            arm["vram_idle_mib"] = vram_used_mib()
            log(f"ready in {ready:.0f}s, VRAM {arm['vram_idle_mib']} MiB")

            for tname, content in (("turn1-prefill", PROMPT),
                                   ("turn2-cache", PROMPT + SUFFIX)):
                try:
                    resp, wall = ask(key, content)
                except Exception as exc:                      # noqa: BLE001
                    arm["turns"].append({"turn": tname, "error": repr(exc)[:400]})
                    log(f"{tname}: ERROR {exc}")
                    break
                tm = resp.get("timings") or {}
                usage = resp.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                rec = {
                    "turn": tname,
                    "wall_s": round(wall, 2),
                    "prompt_n": tm.get("prompt_n") or usage.get("prompt_tokens"),
                    "prompt_tok_s": tm.get("prompt_per_second"),
                    "prompt_ms": tm.get("prompt_ms"),
                    "predicted_n": tm.get("predicted_n") or usage.get("completion_tokens"),
                    "decode_tok_s": tm.get("predicted_per_second"),
                    "cached_tokens": details.get("cached_tokens") or usage.get("cached_tokens"),
                    "vram_after_mib": vram_used_mib(),
                    "answer_head": (resp["choices"][0]["message"]["content"] or "")[:160],
                }
                arm["turns"].append(rec)
                log(f"{tname}: prompt {rec['prompt_n']} tok in {rec['prompt_ms']} ms "
                    f"= {rec['prompt_tok_s']:.0f} tok/s | decode {rec['decode_tok_s']:.1f} tok/s "
                    f"| cached {rec['cached_tokens']} | wall {wall:.1f}s")
        finally:
            arm["engine_log"] = engine_lines(cfg["log"])
            kill_tree(proc)
            srv_log.close()
            time.sleep(4)

        results.append(arm)
        json.dump(results, open(out_json, "w", encoding="utf-8"), indent=1)
        log(f"=== {name} done, {len(results)}/4 arms")

    log("ALL ARMS DONE")


if __name__ == "__main__":
    main()

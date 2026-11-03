"""Arm H: the shape DeepSeek Harness actually sends to a local model.

System prompt + ~a dozen tool schemas (the cost a real agent harness adds),
a ~108K-token user message, then a follow-up turn with the assistant's reply
and a tool result appended. Same engine configuration as arm E/G.
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
RAW = open(os.path.join(OUT, "prompt.txt"), encoding="utf-8").read()
PROMPT = RAW[:330000]                      # ~108K tokens, leaving room for system + tools
PORT = 8096
URL = f"http://127.0.0.1:{PORT}"
CFG_SRC = os.path.join(ROOT, "strata-qwen36-ud-q3_k_m-128k-fast.json")

SYSTEM = (
    "You are a coding agent working inside a repository on the user's machine. "
    "Use the tools to read files before editing them. Prefer targeted edits. "
    "When a command fails, investigate the failure instead of retrying blindly. "
    "Working directory: <HOME>\\Documents\\deepseek-harness\\default-workspace. "
    "Read a file before editing it; use glob to find files; use grep to search contents."
)

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a UTF-8 text file and return line-numbered content.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Create or fully replace a UTF-8 text file.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace literal text in an existing file.",
     "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files whose paths match a glob pattern.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents with a regular expression.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "pwsh", "description": "Execute a PowerShell command and return stdout/stderr.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}, "workdir": {"type": "string"}}, "required": ["command", "description"]}}},
    {"type": "function", "function": {"name": "todo_write", "description": "Record and update a task list.",
     "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object"}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "job_list", "description": "List background jobs with ids and statuses.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "job_output", "description": "Read a background job's output.",
     "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "wait": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "subagent", "description": "Delegate a self-contained task to a subagent.",
     "parameters": {"type": "object", "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["description", "prompt"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for current information.",
     "parameters": {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}}, "required": ["queries"]}}},
    {"type": "function", "function": {"name": "present", "description": "Declare files as final deliverables.",
     "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "object"}}}, "required": ["files"]}}},
]

TURN2 = ("I ran the grep. Result: 3 matches in src/poly/engine.cpp at lines 1046, 1058 and 1066. "
         "Now tell me, in one sentence, what decides which engine class is constructed.")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ask(key, messages, max_tokens=64):
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": "qwen3.6-35b-a3b", "messages": messages, "tools": TOOLS,
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    t0 = time.time()
    r = requests.post(URL + "/v1/chat/completions", headers=h, json=body, timeout=1800)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall


def main():
    cfg = json.load(open(CFG_SRC, encoding="utf-8"))
    cfg["port"] = PORT
    cfg["log"] = os.path.join(OUT, "engine-H-dsh.log")
    cfg_path = os.path.join(OUT, "cfg-H-dsh.json")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=1)
    key = cfg["api_key"]
    srv = open(os.path.join(OUT, "server-H-dsh.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([PY, os.path.join(ROOT, "serve", "server.py"), "--engine", "strata",
                             "--config", cfg_path, "--port", str(PORT)],
                            cwd=ROOT, stdout=srv, stderr=subprocess.STDOUT)
    out = {"arm": "H-dsh-shape", "turns": []}
    try:
        h = {"Authorization": f"Bearer {key}"}
        for _ in range(300):
            try:
                if requests.get(URL + "/v1/models", headers=h, timeout=10).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(3)

        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}]
        resp, wall = ask(key, msgs)
        msg = resp["choices"][0]["message"]
        answer = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        tm = resp.get("timings") or {}
        out["turns"].append({"turn": "1 with system+tools", "prompt_n": tm.get("prompt_n"),
                             "prompt_tok_s": tm.get("prompt_per_second"),
                             "decode_tok_s": tm.get("predicted_per_second"), "wall_s": round(wall, 1),
                             "tool_calls": len(calls), "answer": answer[:200]})
        log(f"turn1 prompt {tm.get('prompt_n')} @ {tm.get('prompt_per_second'):.0f} tok/s, "
            f"calls {len(calls)}, wall {wall:.1f}s")

        msgs2 = msgs + [{"role": "assistant", "content": answer,
                         "tool_calls": calls or None},
                        {"role": "tool", "tool_call_id": (calls[0].get("id") if calls else "x"),
                         "content": "3 matches: engine.cpp:1046, 1058, 1066"},
                        {"role": "user", "content": TURN2}]
        resp, wall = ask(key, msgs2)
        tm = resp.get("timings") or {}
        usage = resp.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        out["turns"].append({"turn": "2 after a tool result", "prompt_n": tm.get("prompt_n"),
                             "prompt_tok_s": tm.get("prompt_per_second"),
                             "decode_tok_s": tm.get("predicted_per_second"),
                             "cached": cached, "wall_s": round(wall, 1)})
        log(f"turn2 read {tm.get('prompt_n')} tok, cached {cached}, wall {wall:.1f}s")
    except Exception as exc:                     # noqa: BLE001
        log(f"ERROR {exc!r}")
        out["error"] = repr(exc)[:500]
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, text=True)
        srv.close()
    json.dump(out, open(os.path.join(OUT, "results-H.json"), "w", encoding="utf-8"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()

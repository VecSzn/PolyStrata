"""Exercise the unchanged native serve protocol with synthetic weights only."""
import queue
import subprocess
import sys
import tempfile
import threading

exe, model = sys.argv[1:3]
extra = sys.argv[3:]
long_prompt = "--long-prompt" in extra
extra = [x for x in extra if x != "--long-prompt"] or ["--no-draft"]
with tempfile.TemporaryFile(mode="w+") as log:
    proc = subprocess.Popen([exe, "--model", model, "--serve", "--max-context", "512" if long_prompt else "128", "--prompt-batch", "32",
                             "--threads", "2", *extra, "--conversation-cache-mib", "16",
                             "--conversation-cache-min-tokens", "1", "--conversation-cache-min-free-mib", "0"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    lines = queue.Queue()
    def reader():
        for line in proc.stdout:
            lines.put(line.rstrip())
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    def until(prefix):
        result = []
        while True:
            line = lines.get(timeout=30)
            assert line is not None, "engine ended before " + prefix
            assert not line.startswith("ERR"), line
            result.append(line)
            if line.startswith(prefix):
                return result
    def request(ids):
        proc.stdin.write("GEN 4 temperature=0 turn_at=3 " + ids + "\n")
        proc.stdin.flush()
        result = until("DONE ")
        tokens = [line for line in result if line.startswith("T ")]
        assert len(tokens) == 4, result
        return tokens
    try:
        until("READY ")
        prompt = ",".join(map(str, range(1, 161 if long_prompt else 65)))  # two 32-token MMQ prompt batches
        first = request(prompt)
        request(",".join(map(str, range(90, 122))))
        again = request(prompt)
        assert first == again, (first, again)
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        assert proc.wait(timeout=30) == 0
        log.seek(0)
        diagnostic = log.read()
        assert "conversation cache: parked" in diagnostic, diagnostic
        assert "conversation cache: restored" in diagnostic, diagnostic
        print("native serve: two conversations, parked/restored state, deterministic replay, clean QUIT")
    except Exception:
        log.seek(0)
        print(log.read(), file=sys.stderr)
        raise
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

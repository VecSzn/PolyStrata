"""Native q36 prompt, graph, MTP, hybrid and parallel-slot regression. Synthetic weights only."""
import queue
import subprocess
import sys
import tempfile
import threading

exe, model = sys.argv[1:3]
base = [exe, '--model', model, '--ctx', '512', '--chunk', '32', '--no-warmup']
prompt = ','.join(map(str, range(1, 65)))
def generate(extra):
    r = subprocess.run(base + ['--tokens', prompt, '--n', '8'] + extra, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert 'gen:' in r.stdout, r.stdout
    return r.stdout.strip()
plain = generate([])
assert generate(['--no-graph']) == plain
assert generate(['--mtp', '2']) == plain
# CPU expert arithmetic uses ggml's activation type; compare modes within that placement.
hybrid = ['--expert-slots', '1', '--pool-workers', '2']
assert generate(hybrid) == generate(hybrid + ['--mtp', '2'])
with tempfile.TemporaryFile(mode='w+') as log:
    proc = subprocess.Popen(base + ['--serve', '--batch', '2', '--batch-mtp', '--mtp', '2'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    lines = queue.Queue()
    def reader():
        for line in proc.stdout:
            lines.put(line.strip())
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    def until(prefix):
        result = []
        while True:
            line = lines.get(timeout=30)
            assert line is not None and not line.startswith('ERR'), line
            result.append(line)
            if line.startswith(prefix):
                return result
    def send(text):
        proc.stdin.write(text + '\n')
        proc.stdin.flush()
    try:
        until('READY ')
        send('GEN 8 temperature=0 ' + prompt)
        a = [s for s in until('DONE ') if s.startswith('T ')]
        assert len(a) == 8, a
        send('GEN 8 temperature=0 ' + ','.join(map(str, range(90, 122))))
        until('DONE ')
        send('GEN 8 temperature=0 ' + prompt)
        b = [s for s in until('DONE ') if s.startswith('T ')]
        assert a == b, (a, b)
        send('BGEN 0 8 temperature=0 ' + prompt + '\nBGEN 1 8 temperature=0 ' + prompt)
        done = set(); tokens = {0: [], 1: []}; first = None
        while len(done) < 2:
            line = lines.get(timeout=30)
            assert line is not None and not line.startswith('ERR'), line
            fields = line.split()
            if fields[0] == 'T':
                first = fields[1]
            if fields[0] == 'BADM':
                assert first is not None
                tokens[int(fields[1])].insert(0, first)
                first = None
            if fields[0] == 'BT':
                tokens[int(fields[1])].append(fields[2])
            if fields[0] == 'BDONE':
                done.add(int(fields[1]))
        assert len(tokens[0]) == len(tokens[1]) == 8, tokens
        assert tokens[0] == tokens[1] == [s.split()[1] for s in a], (tokens, a)
        proc.stdin.close()
        assert proc.wait(timeout=30) == 0
        print('q36: graph/plain/MTP parity, hybrid MTP parity, conversation replay, two parallel MTP slots, clean shutdown')
    except Exception:
        log.seek(0)
        print(log.read(), file=sys.stderr)
        raise
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

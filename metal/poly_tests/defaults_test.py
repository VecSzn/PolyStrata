"""Check effective Metal defaults and overrides with generated GGUFs, not real weights."""
from pathlib import Path
import subprocess
import sys

exe, directory = sys.argv[1:]
directory = Path(directory)


def run(model, *options, expected=None, error=None):
    result = subprocess.run([exe, '--model', str(directory / model), '--tokens', '1,2,3,4,5',
                             '--max-new', '4', '--max-context', '128', '--prompt-batch', '32',
                             '--threads', '2', '--no-draft', *options],
                            capture_output=True, text=True, timeout=60)
    if error:
        assert result.returncode != 0 and error in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert f'Metal {expected}' in result.stderr, result.stderr
    return result.stdout


for arch in ('lfm', 'qwen35moe', 'qwen35moe-fp16', 'gemma4', 'glm5-next', 'deepseek4'):
    model = f'tiny-{arch}.gguf'
    kv = 'q8_0' if arch == 'qwen35moe' else 'fp16'
    expected = f'optimize=on memory=balanced kv={kv}'
    default = run(model, expected=expected)
    explicit = run(model, '--metal-optimize', '--metal-memory', 'balanced', '--kv', kv, expected=expected)
    assert default == explicit, (arch, default, explicit)

run('tiny-qwen35moe.gguf', '--no-metal-optimize', '--metal-memory', 'legacy', '--kv', 'fp16',
    expected='optimize=off memory=legacy kv=fp16')
run('tiny-qwen35moe.gguf', '--metal-optimize', '--no-metal-optimize', '--kv', 'fp16',
    expected='optimize=off memory=balanced kv=fp16')
run('tiny-qwen35moe.gguf', '--no-metal-optimize', '--metal-optimize',
    expected='optimize=on memory=balanced kv=q8_0')
run('tiny-gemma4.gguf', '--kv', 'q8_0', error='only supported for qwen35moe')
run('tiny-qwen35moe-fp16.gguf', '--kv', 'q8_0', error='multiple of 32')
run('tiny-lfm.gguf', '--metal-memory', 'invalid', error='requires legacy or balanced')
print('Metal defaults: effective settings, explicit equivalence, opt-outs and rejected unsupported formats')

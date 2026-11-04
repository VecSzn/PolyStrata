"""Reported-device adapter, not an upstream support-list or engine patch.

Requires the matching official TheRock gfx1010 SDK/runtime installed already.
Requires the pinned GGUF already in workspace/models; may download supporting
packs/build dependencies and compile. Uses private setup APIs at the pinned commit.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--family', choices=['qwen36','ornith','tiel','gemma4'], default='qwen36')
    parser.add_argument('--reuse-tested-engine', action='store_true')
    options = parser.parse_args()
    source, workspace = options.source.resolve(), options.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    quant,filename = {
        'qwen36':('UD-Q3_K_M','Qwen3.6-35B-A3B-UD-Q3_K_M.gguf'),
        'ornith':('APEX-I-COMPACT','Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf'),
        'tiel':('UD-Q3_K_XL','Tiel-Coder-35B-A3B-MTP-UD-Q3_K_XL.gguf'),
        'gemma4':('QAT-UD-Q4_K_XL','gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'),
    }[options.family]
    model = workspace / 'models' / filename
    if not model.is_file():
        parser.error('Place the pinned, SHA256-verified GGUF in workspace/models first')
    os.chdir(source)
    os.environ['PIP_CACHE_DIR'] = str(workspace / 'pip-cache')
    os.environ['PYTHONUNBUFFERED'] = '1'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('poly_upstream_setup', source / 'setup.py')
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    if options.reuse_tested_engine:
        def reuse_engine(gpu,llama):
            engine=source/'engine-poly'
            meta=json.loads((engine/'BUILD.json').read_text())
            if meta.get('backend')!='hip' or meta.get('archs')!=['gfx1010'] or meta.get('src')!=up.source_hash(up.ENGINE_SOURCES):
                raise RuntimeError('Existing engine metadata does not match pinned HIP source/gfx1010')
            with (engine/'strata-poly.exe').open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!='a712397d4e2d82d8c207627766ba991e615ae7fc648d90e6ebfddf4c970ae959':
                raise RuntimeError('Existing executable differs from tested engine')
            return engine
        up.build_engine_poly_hip=reuse_engine
    captured = None
    original_parse = argparse.ArgumentParser.parse_args
    class Captured(BaseException):
        pass
    def capture(parser, *args, **kwargs):
        nonlocal captured
        captured = original_parse(parser, *args, **kwargs)
        raise Captured()
    sys.argv = [str(source / 'setup.py'), '--family', options.family, '--model', quant,
                '--backend', 'hip', '--yes', '--no-start', '--no-browser',
                '--gguf-dir', str(workspace / 'models'), '--data-dir', str(workspace / 'data'),
                '--port', '18085']
    argparse.ArgumentParser.parse_args = capture
    try:
        up.main()
    except Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original_parse
    if captured is None:
        raise RuntimeError('Pinned setup did not expose its expected parser')
    captured._custom_gguf_shards = None
    captured.models_dir = str(workspace / 'data/models')
    gpu = {'name': 'AMD Radeon RX 5700 XT', 'arch': 'gfx1010', 'archs': ['gfx1010'],
           'vram_gb': 8176 / 1024, 'index': 0, 'count': 1, 'driver': 'hip'}
    raise SystemExit(up.poly_setup(captured, options.family, gpu, True, None, None, 256.0,
                                   workspace / 'data', [], 18085, None))

if __name__ == '__main__':
    main()

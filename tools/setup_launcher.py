"""The model -> quantization -> context menu used by setup.sh on macOS and Linux.

Catalogs and installation remain owned by setup.py / metal/setup_mac.py. This launcher only
collects missing choices in a terminal, then replaces itself with the platform's installer.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup as S


def catalog(platform: str, backend: str | None = None) -> dict:
    if platform == 'darwin':
        from metal import setup_mac as M
        result = {}
        for family, fam in M.model_catalog().items():
            result[family] = dict(fam, contexts=[2048, 4096, *M.CONTEXTS],
                                  note='experimental Metal' if family not in ('qwen', 'unsloth') else 'Metal')
        return result
    result = {}
    for family in S.menu_families(backend in ('hip', 'sycl')) + list(S.OWN_PATH_FAMILIES):
        fam = S.OWN_PATH_FAMILIES.get(family) or S.FAMILIES[family]
        contexts = S.POLY_CONTEXTS if fam.get('engine') == 'poly' else S.CONTEXTS
        if family in S.OWN_PATH_FAMILIES:
            contexts = [c for c in contexts if c <= 262144]
        result[family] = dict(fam, models=fam['models'] if family in S.OWN_PATH_FAMILIES else S.family_models(family),
                              contexts=[2048, 4096, *contexts],
                              note='NVIDIA only' if fam.get('ready') else fam.get('by', ''))
    return result


def saved_choices(platform: str, entries: dict) -> list[dict]:
    if platform == 'darwin':
        from metal import setup_mac as M
        configs = M.mac_configs()
    else:
        configs = S.installed_configs()
    saved = []
    for path in configs:
        try:
            choice = S.choices_from_config(path)
            if choice.get('family') in entries and choice.get('model') in entries[choice['family']]['models']:
                saved.append(choice)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return saved


def list_models(platform: str, backend: str | None = None) -> None:
    for family, entry in catalog(platform, backend).items():
        print(f'{family}: {entry["title"]} — {entry["note"]}')
        for model, spec in entry['models'].items():
            size = f" ({spec['download_gb']:.1f} GB download)" if 'download_gb' in spec else ''
            print(f'  {model}{size}')
        print('  Context tokens: ' + ', '.join(str(c) for c in entry['contexts']) + ', or custom')


def pick(title: str, items: list[tuple[str, str]], default: str) -> str:
    print(f'\n{title}')
    keys = [key for key, _ in items]
    for i, (key, label) in enumerate(items, 1):
        print(f'  {i}) {label}' + ('  [default]' if key == default else ''))
    while True:
        answer = input(f'Choose 1-{len(items)} or a name [{keys.index(default) + 1}], q to quit: ').strip()
        if answer.lower() in ('q', 'quit'):
            raise KeyboardInterrupt
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            return keys[int(answer) - 1]
        for key in keys:
            if answer.casefold() == key.casefold():
                return key
        print('Please choose a listed number or name.')


def quant_default(platform: str, family: str, sizes: dict, ram: float) -> str:
    if platform == 'darwin' and family in ('qwen', 'unsloth'):
        from metal import setup_mac as M
        recommended, _ = M.recommend_model(ram)
        if recommended in sizes:
            return recommended
    elif family in S.FAMILIES and not S.FAMILIES[family].get('models'):
        return S.recommended_size(family, list(sizes), ram)
    return min(sizes, key=lambda m: sizes[m].get('download_gb', sizes[m].get('ram_gb', float('inf'))))


def choose(argv: list[str], platform: str, options: argparse.Namespace) -> list[str]:
    entries = catalog(platform, options.backend)
    saved = saved_choices(platform, entries)
    try:
        if platform == 'darwin':
            from metal import setup_mac as M
            ram = M.mac_info()['ram_gb']
        else:
            ram = S.ram_gb()
    except (OSError, KeyError, ValueError):
        ram = 0.0
    family = options.family
    if family is not None and family not in entries:
        raise ValueError(f'{family!r} is not available on this platform. Choose from: {", ".join(entries)}')
    if family is None:
        available = {f: d for f, d in entries.items() if not options.model or options.model in d['models']}
        if not available:
            raise ValueError(f'No available model family has quantization {options.model!r}.')
        default = next((c['family'] for c in saved if c['family'] in available), next(iter(available)))
        family = pick('1. Model', [(f, f"{d['title']} ({f}) — {d['note']}") for f, d in available.items()], default)
    entry = entries[family]
    previous = next((c for c in saved if c['family'] == family), {})
    model = options.model
    if model is not None and model not in entry['models']:
        raise ValueError(f'{family} has no {model!r} quantization. Choose from: {", ".join(entry["models"])}')
    if model is None:
        items = []
        for name, spec in entry['models'].items():
            detail = f" — {spec['download_gb']:.1f} GB download" if 'download_gb' in spec else ''
            if spec.get('ram_gb'):
                detail += f"; ~{spec['ram_gb']} GB RAM"
            if name == previous.get('model'):
                detail += ' (installed)'
            items.append((name, name + detail))
        model = pick('2. Quantization', items, previous.get('model') or quant_default(platform, family, entry['models'], ram))
    context = options.context
    if context is not None and not 16 <= context <= 1048576:
        raise ValueError('--context must be between 16 and 1048576 tokens.')
    if context is None:
        contexts = list(entry['contexts'])
        default_ctx = 8192 if platform == 'darwin' else 32768
        if previous.get('model') == model and previous.get('context'):
            default_ctx = previous['context']
        if default_ctx not in contexts:
            contexts.append(default_ctx)
        print('\nContext is the text the model can retain. Longer contexts use more memory.')
        selected = pick('3. Context', [(str(c), f'{c:,} tokens ({c / 1024:g}K)') for c in sorted(contexts)]
                        + [('custom', 'Custom token count')], str(default_ctx))
        if selected == 'custom':
            while True:
                value = input('Context tokens (16-1048576), q to quit: ').strip()
                if value.lower() in ('q', 'quit'):
                    raise KeyboardInterrupt
                if value.isdigit() and 16 <= int(value) <= 1048576:
                    context = int(value)
                    break
                print('Enter a whole number between 16 and 1048576.')
        else:
            context = int(selected)
    result = list(argv)
    for flag, supplied, value in (('--family', options.family, family), ('--model', options.model, model),
                                  ('--context', options.context, context)):
        if supplied is None:
            result.extend((flag, str(value)))
    print(f'\nSelected: {entry["title"]}, {model}, {context:,} tokens.\n', flush=True)
    return result


def main(argv=None, platform=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    platform = sys.platform if platform is None else platform
    backend_path = ROOT / 'metal' / 'setup_mac.py' if platform == 'darwin' else ROOT / 'setup.py'
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ('family', 'model', 'backend'):
        parser.add_argument('--' + name)
    parser.add_argument('--context', type=int)
    parser.add_argument('--list-models', action='store_true')
    options, _ = parser.parse_known_args(argv)
    if options.list_models:
        list_models(platform, options.backend)
        return 0
    if '--help' in argv or '-h' in argv:
        print('setup.sh: model → quantization → context menus in a terminal.\n'
              '  --list-models  List this platform\'s families, quantizations and contexts without installing.\n'
              '  --yes          Skip the menus and use the installer\'s defaults.\n', flush=True)
    bypass = {'-h', '--help', '--check', '--list-models', '--inspect', '--update', '--rollback-engine', '--yes', '--calibrate'}
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive and not any(a.split('=', 1)[0] in bypass for a in argv) and options.backend != 'sycl':
        try:
            argv = choose(argv, platform, options)
        except (EOFError, KeyboardInterrupt):
            print('\nModel setup cancelled.')
            return 130
        except ValueError as error:
            print(f'Setup: {error}', file=sys.stderr)
            return 2
    # No shell reconstruction/eval: spaces, secrets and literal metacharacters stay individual arguments.
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, str(backend_path), *argv])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

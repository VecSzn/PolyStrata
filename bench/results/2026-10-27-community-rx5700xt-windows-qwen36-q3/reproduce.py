"""Recreate fixed HTTP fixtures; optionally measure an already-running server."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
VOCAB = 'river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz meadow anchor violet thunder'.split()

def fixture(record):
    f = record['fixture']
    rng = random.Random(f['seed'])
    filler = [rng.choice(VOCAB) for _ in range(f['filler_words'])]
    mid = len(filler) // 2
    codes = f['expected']
    prompt = ('Find three exact reference codes in the following archive. Ignore ordinary filler words.\n'
              + 'ALPHA_SENTINEL is ' + codes['alpha'] + '.\n' + ' '.join(filler[:mid])
              + '\nBETA_SENTINEL is ' + codes['beta'] + '.\n' + ' '.join(filler[mid:])
              + '\nGAMMA_SENTINEL is ' + codes['gamma']
              + '.\nReturn only JSON with keys alpha, beta, gamma and their exact reference codes.')
    assert hashlib.sha256(prompt.encode('utf-8')).hexdigest() == f['utf8_lf_sha256']
    return prompt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--context', type=int, choices=[65536, 131072], default=65536)
    parser.add_argument('--url', default='http://127.0.0.1:18085')
    args = parser.parse_args()
    records = [json.loads((ROOT / name).read_text()) for name in ['context64.json', 'context128.json']]
    for record in records:
        fixture(record)
        assert json.loads(record['response']['text']) == record['fixture']['expected']
        assert record['response']['usage']['prompt_tokens'] == record['metrics']['prompt_tokens']
    for check in json.loads((ROOT / 'basic-checks.json').read_text()):
        assert check['passed'] and check['metrics']['ok']
    print('Both fixed HTTP fixtures verified; retained retrieval answers and basic checks pass.')
    if args.verify_only:
        return
    if not args.source:
        parser.error('--source is required for an inference request')
    sys.path.insert(0, str(args.source.resolve() / 'tools'))
    import strata_bench as bench
    health = bench.try_json(args.url, '/health', timeout=5)
    if not health or not health.get('loaded') or health.get('max_context') != args.context:
        parser.error('Server must be loaded with max_context equal to --context')
    if health.get('model') != 'qwen3.6-35b-a3b-ud-q3_k_m':
        parser.error('Server must be running the reported Qwen3.6 UD-Q3_K_M model')
    record = next(r for r in records if r['configured_context'] == args.context)
    response = bench.chat(args.url, '', [{'role': 'user', 'content': fixture(record)}], 128, timeout=1800)
    try:
        passed = not response.get('err') and json.loads(response['text'].strip()) == record['fixture']['expected']
    except (ValueError, KeyError):
        passed = False
    print(json.dumps({'configured_context': args.context, 'metrics': bench.digest(response),
                      'response': response, 'passed': passed}, indent=2))
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()

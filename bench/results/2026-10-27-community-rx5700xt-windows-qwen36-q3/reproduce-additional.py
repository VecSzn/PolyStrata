"""Verify all fixed long fixtures offline, or probe an already-running server."""
import argparse
import json
from pathlib import Path
import sys
from reproduce import fixture

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--model',choices=['ornith','tiel','gemma4'],default='ornith')
    parser.add_argument('--context',type=int,choices=[65536,131072],default=65536)
    parser.add_argument('--url',default='http://127.0.0.1:18085')
    args=parser.parse_args()
    files=list((ROOT/'additional-models').glob('*/context64.json'))+list((ROOT/'additional-models').glob('*/context128.json'))
    for path in files:
        record=json.loads(path.read_bytes());fixture(record)
        assert record['metrics']['prompt_tokens']==record['response']['usage']['prompt_tokens']
        assert record['metrics']['generated_tokens']==record['response']['usage']['completion_tokens']
    print('All additional long-fixture hashes and retained API token counts verified.')
    if args.verify_only:return
    if not args.source:parser.error('--source is required for inference')
    folder=ROOT/'additional-models'/args.model
    context_file=folder/('context64.json' if args.context==65536 else 'context128.json')
    if not context_file.exists():parser.error('That model/context was not tested in this report')
    sys.path.insert(0,str(args.source.resolve()/'tools'))
    import strata_bench as bench
    health=bench.try_json(args.url,'/health',timeout=5)
    expected_model=json.loads((folder/'results.json').read_bytes())['machine']['model']
    if not health or not health.get('loaded') or health.get('max_context')!=args.context or health.get('model')!=expected_model:
        parser.error('Server must be loaded with the reported model and selected context')
    record=json.loads(context_file.read_bytes());prompt=fixture(record);cap=128
    response=bench.chat(args.url,'',[{'role':'user','content':prompt}],cap,timeout=1800)
    try:strict=not response.get('err') and json.loads(response['text'].strip())==record['fixture']['expected']
    except (ValueError,KeyError):strict=False
    facts=sum(code in response.get('text','') for code in record['fixture']['expected'].values())
    print(json.dumps(dict(metrics=bench.digest(response),response=response,strict_json_pass=strict,
                          exact_codes_present=facts),indent=2))

if __name__=='__main__':main()

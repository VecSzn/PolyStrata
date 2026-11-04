"""Verify the retained custom comparison, or send its block to a fresh server.

Start each arm separately with the frozen profile and documented environment.
This script does not start/stop servers or reset profiles between tasks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from reproduce import fixture

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--url',default='http://127.0.0.1:18085')
    parser.add_argument('--initial-profile',type=Path)
    args=parser.parse_args()
    provenance=json.loads((ROOT/'tail-skip/provenance.json').read_bytes())
    tasks=json.loads((ROOT/'tail-skip/tasks.json').read_bytes())
    result=json.loads((ROOT/'tail-skip/results.json').read_bytes())
    assert hashlib.sha256((ROOT/'tail-skip/initial-profile.txt').read_bytes()).hexdigest()==provenance['initial_profile_sha256']
    assert len(result['arms'])==6 and len(tasks)==5
    for task in tasks:
        if task['name']=='long':
            prompt=fixture(task)
            task['request']['messages']=[{'role':'user','content':prompt}]
    for arm in result['arms']:
        assert arm['profiles']['before']==provenance['initial_profile_sha256']
        assert [r['name'] for r in arm['results']]==[t['name'] for t in tasks]
        long=next(r for r in arm['results'] if r['name']=='long')
        assert long['metrics']['prompt_tokens']==58117 and long['metrics']['reused_tokens']==0
        assert long['strict_pass'] and long['facts_exact']
    print('Six retained arms, frozen profile and fixed long HTTP fixture verified.')
    if args.verify_only:return
    if not args.source or not args.initial_profile:parser.error('--source and --initial-profile are required')
    if hashlib.sha256(args.initial_profile.read_bytes()).hexdigest()!=provenance['initial_profile_sha256']:
        parser.error('Initial profile must be a fresh byte-identical copy before this arm')
    sys.path.insert(0,str(args.source.resolve()/'tools'))
    import strata_bench as bench
    health=bench.try_json(args.url,'/health',timeout=5)
    if not health or not health.get('loaded') or health.get('max_context')!=65536 or health.get('model')!='qwen3.6-35b-a3b-ud-q3_k_m':
        parser.error('Fresh server must run the reported Qwen model at65536 context')
    rows=[]
    for task in tasks:
        response=bench.chat(args.url,'',task['request']['messages'],task['cap'],timeout=900)
        rows.append(dict(name=task['name'],response=response,metrics=bench.digest(response)))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()

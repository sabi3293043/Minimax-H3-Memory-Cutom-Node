"""Compare two opt-in JSON reports; does not queue jobs or score video quality."""
import argparse
import json
from pathlib import Path


def flatten(value, prefix=''):
    result={}
    for key,item in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):result.update(flatten(item,name))
        else:result[name]=item
    return result


def compare(left,right):
    if any(r.get('schema')!=1 or not isinstance(r.get('settings'),dict)
           or not isinstance(r.get('metrics'),dict) for r in (left,right)):
        raise ValueError('Both files must be schema=1 MiniMax Memory JSON reports')
    a,b=flatten(left['settings']),flatten(right['settings'])
    changes={key:[a.get(key),b.get(key)] for key in sorted(a.keys()|b.keys()) if a.get(key)!=b.get(key)}
    return dict(same_seed=left.get('seed')==right.get('seed'),changed_settings=changes,
        runs=[dict(run_id=r.get('run_id'),metrics=r.get('metrics'),error=r.get('error')) for r in (left,right)],
        limitation='Reports do not verify checkpoint/LoRA file contents or measure identity, clothing or video quality. Keep the workflow fixed and compare the videos.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left',type=Path);parser.add_argument('right',type=Path)
    args=parser.parse_args()
    print(json.dumps(compare(json.loads(args.left.read_text()),json.loads(args.right.read_text())),indent=2))

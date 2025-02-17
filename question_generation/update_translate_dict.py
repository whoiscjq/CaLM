import sys
import json
import os
import argparse

def update_translate_dict(dict_file, input_file):
    with open(dict_file) as f:
        Dict = json.load(f)
    lines = open(input_file).readlines()
    
    for line in lines:
        obj = json.loads(line)
        try:
            result = json.loads(obj["response"]["choices"][0]["message"]["content"])
            assert len(result) == 5
            for item in result:
                assert type(item) == str and len(item) > 0
            if obj["node_name"] not in Dict:
                Dict[obj["node_name"]] = result
        except Exception as e:
            print(e)
            continue
    
    with open(dict_file, 'w') as f_out:
        f_out.write(json.dumps(Dict, indent='  ', ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dict_file",
        default='data/translate_dict.json',
        type=str
    )
    parser.add_argument(
        "--input_file",
        type=str,
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    update_translate_dict(
        args.dict_file,
        args.input_file
    )
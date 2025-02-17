import sys
import json
import os
import argparse

def update_relation_dict(dict_file, input_file):
    try:
        with open(dict_file) as f:
            Dict = json.load(f)
    except:
        Dict = {}

    lines = open(input_file).readlines()
    
    for line in lines:
        obj = json.loads(line)
        node_name_A, node_name_B = obj["node_name_A"], obj["node_name_B"]
        key = f"{node_name_A}#{node_name_B}"
        if key in Dict: continue
        try:
            result = obj["response"]["choices"][0]["message"]["content"]
            assert result == "1" or result == "-1"
            Dict[key] = int(result)
        except Exception as e:
            print(e)
            continue
    
    with open(dict_file, 'w') as f_out:
        f_out.write(json.dumps(Dict, indent='  ', ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dict_file",
        default='data/relation_dict.json',
        type=str
    )
    parser.add_argument(
        "--input_file",
        type=str,
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    update_relation_dict(
        args.dict_file,
        args.input_file
    )
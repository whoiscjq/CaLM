import sys
import json
import os
import random
import argparse

def gen_relation_prompt(dict_file, input_file, output_file):
    dirname = os.path.dirname(output_file)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    f_out = open(output_file, 'w')

    with open(dict_file) as f:
        Dict = json.load(f)

    lines = open(input_file).readlines()
    
    for line in lines:
        obj = json.loads(line)
        try:
            result = json.loads(obj["response"]["choices"][0]["message"]["content"])
        except Exception as e:
            print(e)
            continue
        
        for edge in obj["edge"]:
            node_A, node_B = edge.split('->')
            node_name_A, node_name_B = result[node_A], result[node_B]
            if not node_name_A or type(node_name_A) != str or len(node_name_A) == 0: continue
            if not node_name_B or type(node_name_B) != str or len(node_name_B) == 0: continue
            node_name_A = node_name_A.lower()
            node_name_B = node_name_B.lower()
            assert node_name_A in Dict, f"{node_name_A} not in Dict" 
            assert node_name_B in Dict, f"{node_name_B} not in Dict"
            prompt = "Given two short sentences, if the first is positively related to the second, then you should return 1. Otherwise if they are negatively related, then you should return -1.\n"

            prompt += "Here are some examples:\n"

            prompt += 'Input: ["education level is low", "income level is high"] Output: -1\n'
            prompt += 'Input: ["education level is low", "income level is low"] Output: 1\n'
            prompt += 'Input: ["education level is high", "income level is high"] Output: 1\n'
            prompt += 'Input: ["education level is high", "income level is low"] Output: -1\n'
            prompt += 'Input: ["the amount of rainfall is low", "the soil moisture level is dry"] Output: 1\n'
            prompt += 'Input: ["the amount of rainfall is low", "the soil moisture level is moist"] Output: -1\n'
            prompt += 'Input: ["the amount of rainfall is high", "the soil moisture level is dry"] Output: -1\n'
            prompt += 'Input: ["the amount of rainfall is high", "the soil moisture level is moist"] Output: 1\n'

            prompt += f'Input: ["{node_name_A} is {Dict[node_name_A][1]}", "{node_name_B} is {Dict[node_name_B][1]}"], Output:'

            f_out.write(json.dumps({"prompt": prompt, "node_name_A": node_name_A, "node_name_B": node_name_B}, ensure_ascii=False) + '\n')

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
    parser.add_argument(
        "--output_file",
        type=str,
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gen_relation_prompt(
        args.dict_file,
        args.input_file,
        args.output_file
    )
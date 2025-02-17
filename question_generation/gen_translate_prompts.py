import sys
import json
import os
import random
import argparse
    

def gen_translate_prompt(dict_file, input_file, output_file, example_num):
    dirname = os.path.dirname(output_file)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    f_out = open(output_file, 'w')

    with open(dict_file) as f:
        Dict = json.load(f)

    lines = open(input_file).readlines()
    
    
    for line in lines:
        obj = json.loads(line)
        print(type(obj["response"]["choices"][0]["message"]["content"]))
        try:
            result = json.loads(obj["response"]["choices"][0]["message"]["content"])
        except Exception as e:
            print(e)
            continue
        
        for key in result:
            node_name = result[key]
            if not node_name or type(node_name) != str or len(node_name) == 0: continue 
            node_name = node_name.lower()
            if node_name in Dict: continue
            prompt = "For a noun or phrase, give the negative and positive adjectives that describe it, and give the Chinese translation of the noun/phrase and the two adjectives. Please use List to present the result and make sure the 5 items are arranged in order."

            if example_num > 0:
                prompt += "\nHere are some examples:\n"
                select_keys = random.sample(Dict.keys(), example_num)
                for select_key in select_keys:
                    s = json.dumps(Dict[select_key], ensure_ascii=False)
                    prompt += f'For "{select_key}", the result List of 5 items is {s}\n'
            
            prompt += f'Now for "{node_name}", what is the result List?'

            f_out.write(json.dumps({"prompt": prompt, "node_name": node_name}, ensure_ascii=False) + '\n')

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
    parser.add_argument(
        "--example_num",
        type=int,
        default=3
    )
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    gen_translate_prompt(
        args.dict_file,
        args.input_file,
        args.output_file,
        args.example_num
    )

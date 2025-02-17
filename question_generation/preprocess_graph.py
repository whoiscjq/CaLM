import sys
import json
import os
import argparse


def preprocess_graph(dict_file, relation_dict_file, input_file, output_file, add):
    with open(dict_file) as f:
        Dict = json.load(f)
    
    with open(relation_dict_file) as f:
        Relation_Dict = json.load(f)

    lines = open(input_file).readlines()
    
    if add:
        try:
            with open(output_file) as f:
                all_results = json.load(f)
        except:
            all_results = []
    else:
        all_results = []
    for line in lines:
        obj = json.loads(line)
        nodes, edges = obj["node"], obj["edge"]
        try:
            result = json.loads(obj["response"]["choices"][0]["message"]["content"])
        except Exception as e:
            print(e)
            continue
        
        flag = True
        for node in nodes:
            if node not in result or type(result[node]) != str or len(result[node]) == 0:
                flag = False
                break
        if not flag:
            continue

        node_name = {}
        edge_bias = {}
        node_value = {}
        node_name_CN = {}
        node_value_CN = {}
        for node in nodes:
            key = result[node].lower()
            assert key in Dict
            value = Dict[key]
            node_name[node] = key
            node_value[node] = [value[0], value[1]]
            node_name_CN[node] = value[2]
            node_value_CN[node] = [value[3], value[4]]

        for edge in edges:
            x, y = edge.split('->')
            node_name_x, node_name_y = node_name[x], node_name[y]
            key = f"{node_name_x}#{node_name_y}"
            assert key in Relation_Dict, f"{key} not in relation dict"
            edge_bias[edge] = {
                "str":  node_name_x + ' -> ' + node_name_y,
                "value": Relation_Dict[key]
            }
        all_results.append({
            "nodes": nodes,
            "edges": edges,
            "node_name": node_name,
            "node_name_CN": node_name_CN,
            "node_value": node_value,
            "node_value_CN": node_value_CN,
            "edge_sign": edge_bias,
        })
    with open(output_file, 'w') as f_out:
        f_out.write(json.dumps(all_results, indent="  ", ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dict_file",
        default='data/translate_dict.json',
        type=str
    )
    parser.add_argument(
        "--relation_dict_file",
        default="data/relation_dict.json",
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
        "--add",
        action="store_true",
        default=False
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preprocess_graph(args.dict_file, args.relation_dict_file, args.input_file, args.output_file, args.add)
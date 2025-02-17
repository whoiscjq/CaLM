import yaml
import argparse
import random
import json
import os


def parse_graph(line):
    line = line.replace('(', '[')
    line = line.replace(')', ']')
    data = json.loads(line)
    n = 0
    for item in data:
        n = max(n, item[0])
        n = max(n, item[1])
    n += 1
    nodes = [chr(ord('A') + i ) for i in range(n)]
    edges = []
    for item in data:
        edges.append(chr(ord('A') + item[0]) + '->' + chr(ord('A') + item[1]))
    return nodes, edges


def gen_prompt(example_num, example_file, graph_file, output_file, repeat_num=1):
    data = open(example_file).read()
    obj = yaml.safe_load(data)

    lines = open(graph_file).readlines()
    f_out = open(output_file, 'w')
    for N in range(repeat_num):
        for line in lines:
            cur_node, cur_edge = parse_graph(line)
            prompt = "For a given pre-defined causal graph with N nodes, can you please assign real-world meaning to each node and make the entire causal graph plausible? Please use json format to present the result."

            if example_num > 0:
                prompt += '\nHere are some examples:\n'
                keys = list(obj["story_list"].keys())
                # select_keys = random.choices(keys, k = example_num)
                select_keys = random.sample(keys, example_num)
                for key in select_keys:
                    story = obj["story_list"][key]
                    prompt += "For a causal graph with {:d} nodes {} and edges {}, we can assign real-world meaning to each node as follows: \n".format(
                        len(story["node"]),
                        ", ".join(story["node"]),
                        ", ".join(story["edge"])
                    )
                    tmp_dict = {}
                    for node in story["node"]:
                        tmp_dict[node] = story["semantic"][node]
                    prompt += json.dumps(tmp_dict, indent='  ')
                    prompt += '\n'
            
            prompt += 'Now, For a causal graph with {:d} nodes {} and edges {}, can you assign real-world meaning to each node? Remember to use json format to answer.'.format(
                len(cur_node),
                ", ".join(cur_node),
                ', '.join(cur_edge)
            )

            f_out.write(json.dumps({
                "prompt": prompt,
                "node": cur_node,
                "edge": cur_edge,
            }) + '\n')
            
            prompt += 'Remember, you should only output answer in json format'
    f_out.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_num",
        default=0,
        type=int,
        help="example number",
    )
    parser.add_argument(
        "--example_file",
        type=str,
        default='config.yaml',
        help="The example file",
    )
    parser.add_argument(
        "--graph_file",
        type=str,
        default='data/graphs/4_nodes.txt'
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='data/gpts/4_nodes_prompt.jsonl'
    )
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=1
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dirname = os.path.dirname(args.output_file)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    gen_prompt(
        example_num=args.example_num,
        example_file=args.example_file,
        graph_file=args.graph_file,
        output_file=args.output_file,
        repeat_num=args.repeat_num
    )

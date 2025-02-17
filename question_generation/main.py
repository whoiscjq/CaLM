import yaml
import json
import argparse
import os

import template
from graph import Graph

def get_background(graph, story):
    background = "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: "
    for node in graph.graph:
        if graph.graph[node]['parent']:
            for parent in graph.graph[node]['parent']:
                s = "{} has a direct effect on {}. ".format(
                    story['semantic'][parent] or 'unobserved confounders',
                    story['semantic'][node] or 'unobserved confounders'
                )
                s = s[0].upper() + s[1:]
                background += s
    return background

def get_data_string(graph, story):
    string = "We define the symbols as follows:\n"
    for node in story['node']:
        for P in [0, 1]:
            assert node + str(P) in story['semantic']
            string += '{}={:d} represents {}.\n'.format(
                node,
                P,
                story['semantic'][node + str(P)]
            )
    string += 'The statistics are as follows:\n'
    string += str(story["data"])
    return string


def generate_question(input_file, output_file, save_invalid, types):
    data = open(input_file).read()
    fout = open(output_file, 'w')
    obj = yaml.safe_load(data)
    filter_invalid = not save_invalid

    all_results = []
    for name in obj["story_list"]:
        story = obj["story_list"][name]
        graph = Graph(story["node"], story["edge"])
        # print(name, '|', graph)
        generate_function = getattr(template, story["type"])
        results = generate_function(
            graph=graph,
            story=story,
            filter_invalid=filter_invalid,
            types=types
        )
        if story.get("background", None):
            background = story["background"]
        else:
            background = get_background(graph, story)
        
        if story.get("data", None):
            data_string = get_data_string(data, story)
        else:
            data_string = ''

        all_results.append({
            "name": name,
            "type": story["type"],
            "background": background,
            "data": data_string,
            "questions": results
        })
    fout.write(json.dumps(all_results, indent='  '))

    fout.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default='config.yaml',
        help="The input config file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='data/questions/output.json',
        help="The output json file with generated questions",
    )
    parser.add_argument(
        "--save_invalid",
        action="store_true",
        help="Save the invalid questions. The groundtruth answer for invalid questions will be \"invalid\". ",
    )
    parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        default=["judge"],
        help="What types of questions to generate. Supported type: judge (yes/no questions), general (general questions)",
    )
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()

    dirname = os.path.dirname(args.output_file)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    
    generate_question(
        input_file=args.input_file,
        output_file=args.output_file,
        save_invalid=args.save_invalid,
        types=args.types
    )


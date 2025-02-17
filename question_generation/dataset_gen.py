import json

base="""Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:"""

def get_problem(item, prompt_style_str=""):
    prompt = prompt_style_str + base % (item["given_info"], item["Background"]["data_info"],item["Instruction"],item["Question"])
    return prompt

def get_solution(item):
    
    word_for_replace="The real world meaning of each node is defined as follows:"
    gt_answer=item.get('gt_answer')
    steps=[]
    if control[0]:
        step0=f"Use symbols to represent each variable: {item['Background']['real_world_meaning'].replace(word_for_replace,'')}"
        steps.append(step0)
    if control[1]:
        step1=f"Abstract a causal graph from the given input info: There is{item['Background']['graph'].replace('Given','')}"
        steps.append(step1)
    if control[2] and item['Background'].get('data_info_math'):
        step2=f"Abstract causal relations of variables from Input Info with math equations: {item['Background']['data_info_math']}"
        steps.append(step2)

    for k,v in item['Reason'].items():
        steps.append(v)
    
    if control[3] and gt_answer:
        steps.append(f'{{"PROB":"{gt_answer}"}}')
    
    labels=["+"]*len(steps)
    return steps,labels,gt_answer

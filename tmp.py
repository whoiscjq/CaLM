import json

data=[]
with open('/mnt/workspace/o1_like/CaLM/calm_lite_dataset/counterfactual/natural_direct_effect/NDE-P_NDE-basic_EN.json', 'r') as file:
    for line in file:
        # Parse each line as a JSON object
        data.append(json.loads(line))
gts=[]
for item in data:
    print(item)
    gts.append(item["gt_answer"])
with open("tmp.txt",'w') as f:
    for idx,gt in enumerate(gts):
        f.write(f"{idx},{gt} \n")

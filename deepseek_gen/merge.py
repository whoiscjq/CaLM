import json

with open("/mnt/workspace/o1_like/CaLM/deepseek_gen/train_output_0.json","r") as f:
    dataset1=json.load(f)
    
with open("/mnt/workspace/o1_like/CaLM/deepseek_gen/train_output_1.json","r") as f:
    dataset2=json.load(f)

dataset=[]
count=0
for idx,item in enumerate(dataset1):
    answer1=item.get("answers")
    answer2=dataset2[idx].get("answers")
    if answer1 and answer2:
        answers=answer1+answer2
    elif answer1:
        answers=answer1
    elif answer2:
        answers=answer2
    else:
        answers=[]
        count=count+1
    item["answers"]=answers
    dataset.append(item)
    if len(answers):
        print(len(answers))
print(count)
with open("total_answer.json","w") as f:
    json.dump(dataset,f)

new_dataset=[]
num_answers=5
global_id=0

for item in dataset:
    answers=item.get("answers")
    del item["answers"]
    if answers and len(answers)>=num_answers:
        for idx,answer in enumerate(answers):
            new_item=item.copy()
            new_item["answer"]=answer
            new_item["dup_id"]=idx
            new_item["global_id"]=global_id
            global_id+=1            
            new_dataset.append(new_item)
    elif answers:
        last_idx=0
        for idx,answer in enumerate(answers):
            new_item=item.copy()
            new_item["answer"]=answer
            new_item["dup_id"]=idx
            new_item["global_id"]=global_id
            global_id+=1 
            new_dataset.append(new_item)
            last_idx=idx
        last_idx+=1
        while last_idx<num_answers-1:
            new_item=item.copy()
            new_item["answer"]=""            
            new_item["dup_id"]=last_idx
            new_item["global_id"]=global_id
            global_id+=1 
            last_idx+=1
            new_dataset.append(new_item)
    else:
        last_idx=0
        while last_idx<num_answers-1:
            new_item=item.copy()
            new_item["answer"]=""            
            new_item["dup_id"]=last_idx
            new_item["global_id"]=global_id
            global_id+=1 
            last_idx+=1
            new_dataset.append(new_item)
            
with open(f"total_train_dup_{num_answers}.json","w") as f:
    json.dump(new_dataset,f)
    
with open(f"total_train_dup_{num_answers}.jsonl","w") as f:
    for item in new_dataset:
        f.write(json.dumps(item)+"\n")
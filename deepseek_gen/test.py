
# import json
# import os
# import time
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
# from tqdm import tqdm
# import requests
# from openai import OpenAI

# def get_response(input):
#     Baseurl = "https://api.claudeshop.top"
#     Skey = "sk-K2dKtLK3O5LlkYr1vcEQ9Ab7SSONSmgB8fCbIJriXm7Of8Vz"
#     payload = json.dumps({
#        "model": "deepseek-r1",
#        "messages": [
#           {
#              "role": "system",
#              "content": "You need to rephrase the following 'process',while keeping each step do the same thing。You should split all steps with special token 'ки', do not split steps with any other way.Always start from 'Step 1:',do not add anything else at the beginning."
#           },
#           {
#              "role": "user",
#              "content": input
#           }
#        ]
#     })
#     url = Baseurl + "/v1/chat/completions"
#     headers = {
#        'Accept': 'application/json',
#        'Authorization': f'Bearer {Skey}',
#        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
#        'Content-Type': 'application/json'
#     }

#     response = requests.request("POST", url, headers=headers, data=payload)

#     # 解析 JSON 数据为 Python 字典
#     data = response.json()

#     # 获取 content 字段的值
#     print(data)
#     return response

# print(get_response("hello, tell me the integral of \frac{1}{x^2+x+1}"))


# import os
# from openai import OpenAI

# client = OpenAI(
#     api_key = os.environ.get("ARK_API_KEY"),
#     base_url = "https://ark.cn-beijing.volces.com/api/v3",
# )

# # Non-streaming:
# print("----- standard request -----")
# completion = client.chat.completions.create(
#     model = "ep-20250220000118-84hmh",  # your model endpoint ID
#     messages = [
#         {"role": "system", "content": "你是人工智能助手"},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
# )
# print(completion.choices[0].message.content)

# import jsonlines
# import sys
# def _load_processed_ids():
#     try:
#         with open('answer_processed.log', 'r') as f:
#             return set(f.read().splitlines())
#     except FileNotFoundError:
#         return set()
# set_aa=_load_processed_ids()
# set_a=set(map(int,set_aa))
# set_b=[]
# print(set_a.pop())
# print(type(set_a.pop()))
# with jsonlines.open("/mnt/workspace/o1_like/CaLM/deepseek_gen/total_train_dup_5.jsonl") as reader:
#     futures = []
#     for item in reader:
#         # print(item["global_id"])
#         # print(type(item["global_id"]))
#         #sys.exit()
#         if item["global_id"] in set_a:
#             print(item["global_id"])
# #print(set_a)
import requests
def _get_response(item):
    Baseurl = "https://api.claudeshop.top"
    Skey = "sk-K2dKtLK3O5LlkYr1vcEQ9Ab7SSONSmgB8fCbIJriXm7Of8Vz"
    question = item["question"]
    correct_answer = item["steps"]
    payload = json.dumps({
    "model": "gpt-4o-mini",
    "messages":[{"role":"system","content":"I want you to act as a professional answer judge. You will be given a step-by-step process and a ground truth answer. Your task is to judge whether the step-by-step process leads to the ground truth answer, slight numerical differences are allowed. Your answer should be a single word: yes or no."},{"role": "user", "content":f"process: {question}\n\n ground_truth: {correct_answer}"}],
    })
    url = Baseurl + "/v1/chat/completions"
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {Skey}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response)

    # 解析 JSON 数据为 Python 字典
    data = response.json()

    # 获取 content 字段的值
    return data['choices'][0]['message']['content']
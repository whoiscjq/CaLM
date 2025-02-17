
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
import requests
from openai import OpenAI

def get_response(input):
    Baseurl = "https://api.claudeshop.top"
    Skey = "sk-K2dKtLK3O5LlkYr1vcEQ9Ab7SSONSmgB8fCbIJriXm7Of8Vz"
    payload = json.dumps({
       "model": "deepseek-r1",
       "messages": [
          {
             "role": "system",
             "content": "You need to rephrase the following 'process',while keeping each step do the same thing。You should split all steps with special token 'ки', do not split steps with any other way.Always start from 'Step 1:',do not add anything else at the beginning."
          },
          {
             "role": "user",
             "content": input
          }
       ]
    })
    url = Baseurl + "/v1/chat/completions"
    headers = {
       'Accept': 'application/json',
       'Authorization': f'Bearer {Skey}',
       'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
       'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # 解析 JSON 数据为 Python 字典
    data = response.json()

    # 获取 content 字段的值
    print(data)
    return data['choices'][0]['message']['content']

print(get_response("hello"))
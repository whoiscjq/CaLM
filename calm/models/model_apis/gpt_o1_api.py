import requests
import json
# 使用中转链接可以和特定的API可以不必向openai发起请求，且请求无须魔法
# 调用方式与openai官网一致，仅需修改baseurl
import time
MODEL = "o1-mini"

def startup(api_key):
    return api_key


def query(context, query_text, dry_run=False):
    
    Baseurl = "https://api.claudeshop.top"
    Skey = context
    payload = json.dumps({
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": query_text
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

    
    return data['choices'][0]['message']['content']




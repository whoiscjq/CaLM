
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
import requests
from openai import OpenAI


# 请根据实际情况实现这个函数
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

def process_dataset(
    input_file: str,
    output_file: str,
    num_answers: int = 10,
    resume: bool = False,
    request_interval: float = 1.0
):
    """
    单线程版本处理函数
    
    参数变化：
    - 移除 max_workers 参数
    - 保持其他参数不变
    """
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 初始化进度数据
    existing_data = []
    if resume and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"发现已有进度，已处理 {len(existing_data)} 条问题")
        except:
            print("警告：无法读取现有进度文件，将重新开始")

    # 准备任务队列
    tasks = []
    for idx, item in enumerate(original_data):
        # 跳过已完整回答的问题
        if idx < len(existing_data) and len(existing_data[idx].get('answers', [])) >= num_answers:
            original_data[idx]['answers'] = existing_data[idx]['answers']
            continue
        
        # 初始化或合并回答
        existing_answers = existing_data[idx]['answers'] if idx < len(existing_data) else []
        original_data[idx]['answers'] = existing_answers.copy()
        
        # 添加需要补充的任务
        remaining = num_answers - len(existing_answers)
        if remaining > 0:
            tasks.extend([(item['question'], idx)] * remaining)

    if not tasks:
        print("所有问题已回答完成，无需继续处理")
        return

    # 初始化进度条
    pbar = tqdm(total=len(tasks), desc="生成回答")
    last_request_time = time.time()  # 初始化请求时间记录

    # 处理任务队列
    for task in tasks:
        question, idx = task
        
        # 速率控制
        elapsed = time.time() - last_request_time
        if elapsed < request_interval:
            sleep_time = request_interval - elapsed
            time.sleep(sleep_time)
        
        # 执行请求
        try:
            response = get_response(question)
            error = None
        except Exception as e:
            response = None
            error = str(e)
        
        # 更新请求时间
        last_request_time = time.time()
        
        # 处理结果
        if response is not None:
            answers = original_data[idx].get('answers', [])
            if len(answers) < num_answers:
                answers.append(response)
                original_data[idx]['answers'] = answers
        
        # 更新进度
        pbar.update(1)
        
        # 错误处理
        if error:
            pbar.write(f"错误：问题索引 {idx} - {error}")
        
        # 定期保存（每10次请求）
        if pbar.n % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, indent=2, ensure_ascii=False)

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)
    pbar.close()
    print("处理完成！")

if __name__ == "__main__":
    process_dataset(
        input_file="dataset/NDE/train.json",
        output_file="deepseek_nde_output0.json",
        num_answers=5,
        resume=True,
        request_interval=0.5
    )
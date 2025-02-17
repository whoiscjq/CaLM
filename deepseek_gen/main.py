
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# 新增重试装饰器（放在函数定义前）
@retry(
    stop=stop_after_attempt(5),  # 最大重试5次
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 指数退避策略
    retry=retry_if_exception_type((requests.exceptions.Timeout, KeyError))  # 针对超时和KeyError重试
)

# 请根据实际情况实现这个函数
@retry(
    stop=stop_after_attempt(5),  # 最大重试5次
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 指数退避策略
    retry=retry_if_exception_type((requests.exceptions.Timeout, KeyError))  # 针对超时和KeyError重试
)
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
       'Content-Type': 'application/json',
        
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=None)  # 建议设置明确超时
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        # 主动抛出API返回的Timeout错误
        if 'error' in data and data['error'].get('code') == 'Timeout':
            raise requests.exceptions.Timeout("API返回超时错误") from e
        raise  # 重新抛出其他异常


# def get_response(query_text, dry_run=False):
#     print(type(query_text))
#     client = OpenAI(api_key ="sk-78099912567f4577beeca9967a0ddd94",base_url="https://api.deepseek.com")

#     response = client.chat.completions.create(
#         model="deepseek-reasoner",
#         messages=[{"role": "user", "content": query_text}],
#         temperature=0,
#     )
#     print(response)
#     return response.choices[0].message.content


def process_dataset(
    input_file: str,
    output_file: str,
    num_answers: int = 10,
    max_workers: int = 4,
    resume: bool = False,
    request_interval: float = 1
):
    """
    处理数据集的主函数
    
    参数：
    input_file: 输入JSON文件路径
    output_file: 输出JSON文件路径
    num_answers: 每个问题需要生成的回答数量
    max_workers: 最大并行工作线程数
    resume: 是否从断点恢复
    request_interval: 请求之间的最小间隔（秒）
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

    # 准备任务列表
    tasks = []
    for idx, item in enumerate(original_data):
        # 获取已有回答
        existing_answers = []
        if idx < len(existing_data) and 'answers' in existing_data[idx]:
            existing_answers = existing_data[idx]['answers']
        
        # 计算需要补充的回答数量
        remaining = num_answers - len(existing_answers)
        if remaining > 0:
            tasks.extend([(item['question'], idx)] * remaining)
    
    if not tasks:
        print("所有问题已回答完成，无需继续处理")
        return

    # 初始化进度条
    pbar = tqdm(total=len(tasks), desc="生成回答")

    # 共享状态
    last_request_time = time.time()
    rate_lock = Lock()
    data_lock = Lock()

    # 包装函数（含速率控制）
    def wrapper(args):
        nonlocal last_request_time
        question, idx = args
        
        with rate_lock:
            # 计算等待时间
            elapsed = time.time() - last_request_time
            if elapsed < request_interval:
                sleep_time = request_interval - elapsed
                time.sleep(sleep_time)
            # response = get_response(question)
            # error=None
            #     error = None
            # 执行请求
            try:
                response = get_response(question)
                error = None
            except Exception as e:
                response = None
                error = str(e)
            
            # 更新时间戳（无论成功失败）
            last_request_time = time.time()
        
        return idx, response, error

    # 结果处理回调
    def handle_result(future):
        idx, response, error = future.result()
        
        with data_lock:
            # 确保不超过最大回答数量
            if response is not None:
                answers = original_data[idx].setdefault('answers', [])
                if len(answers) < num_answers:
                    answers.append(response)
            
            # 定期保存进度
            if pbar.n % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, indent=2, ensure_ascii=False)
        
        pbar.update(1)
        if error:
            pbar.write(f"错误：问题索引 {idx} - {error}")

    # 执行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(wrapper, task)
            future.add_done_callback(handle_result)
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)
    pbar.close()
    print("处理完成！")



# 使用示例（恢复模式）
if __name__=="__main__":
    def gen_input_file(original_file,output_file,dp_time=10):
        result=[]
        with open(original_file,"r") as f:
            data=json.load(f)
        for item in data:
            for idx in range(dp_time):
                tmp=item
                tmp["idx_for_dp"]=idx
                result.append(tmp)
        with open(output_file,"w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
       
    
    process_dataset(
        input_file="/mnt/workspace/o1_like/CaLM/deepseek_gen/dataset/NDE/train.json",
        output_file="train_output1.json",
        num_answers=1,
        max_workers=8,
        resume=True  # 启用断点续传
    )



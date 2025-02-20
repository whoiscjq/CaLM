import json
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((requests.exceptions.Timeout, KeyError, requests.exceptions.ConnectionError))
)
def get_response(input_text):
    """API请求函数（含安全访问和错误处理）"""
    base_url = "https://api.claudeshop.top"
    api_key = "sk-K2dKtLK3O5LlkYr1vcEQ9Ab7SSONSmgB8fCbIJriXm7Of8Vz"
    
    payload = json.dumps({
        "model": "deepseek-r1",
        "messages": [
            {"role": "user", "content": input_text}
        ]
    })

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            data=payload,
            # timeout=(3.05, 10)  # 连接3秒/读取10秒超时
        )
        response.raise_for_status()
        data = response.json()

        # 安全访问响应字段
        if choices := data.get('choices'):
            if message := choices[0].get('message'):
                return message.get('content', '')
        raise KeyError("Invalid API response structure")
        
    except requests.exceptions.JSONDecodeError:
        raise ValueError("Invalid JSON response")
    except Exception as e:
        if hasattr(e, 'response') and e.response:
            error_data = e.response.json()
            if error_data.get('error', {}).get('code') == 'Timeout':
                raise requests.exceptions.Timeout("API timeout") from e
        raise

def process_dataset(
    input_file: str,
    output_file: str,
    num_answers: int = 10,
    max_workers: int = 4,
    resume: bool = False,
    request_interval: float = 2
):
    """数据集处理主函数"""
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 断点续传逻辑
    existing_data = []
    if resume and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"Resuming from {len(existing_data)} processed items")
        except json.JSONDecodeError:
            print("Warning: Corrupted progress file, starting fresh")

    # 任务队列初始化
    tasks = []
    for idx, item in enumerate(original_data):
        existing_answers = existing_data[idx]['answers'] if idx < len(existing_data) and existing_data[idx].get('answers') else []
        remaining = max(0, num_answers - len(existing_answers))
        tasks.extend([(item['question'], idx)] * remaining)

    if not tasks:
        print("All answers already generated")
        return

    # 并发控制参数
    pbar = tqdm(total=len(tasks), desc="Generating Answers")
    last_request_time = time.time()
    rate_lock = Lock()
    data_lock = Lock()

    def api_wrapper(args):
        """带速率控制的请求包装器"""
        nonlocal last_request_time
        question, idx = args

        with rate_lock:
            elapsed = time.time() - last_request_time
            if elapsed < request_interval:
                print(time.time())
                sleep_duration = request_interval - elapsed + random.uniform(0, 2)
                time.sleep(sleep_duration)
                

            try:
                response = get_response(question)
                error = None
            except Exception as e:
                response = None
                error = str(e)
            
            last_request_time = time.time()
            return idx, response, error

    def result_handler(future):
        """结果处理回调"""
        idx, response, error = future.result()
        
        with data_lock:
            if response and len(original_data[idx].get('answers', [])) < num_answers:
                original_data[idx].setdefault('answers', []).append(response)
            
            # 定期保存进度
            if pbar.n % 5 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, indent=2, ensure_ascii=False)
        
        pbar.update(1)
        if error:
            pbar.write(f"Error[{idx}]: {error}")

    # 执行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(api_wrapper, task) for task in tasks]
        for future in futures:
            future.add_done_callback(result_handler)
        
        # 等待完成
        for future in futures:
            future.result()

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)
    pbar.close()
    print("Processing completed")

if __name__ == "__main__":
    process_dataset(
        input_file="/mnt/workspace/o1_like/CaLM/deepseek_gen/dataset/NDE/train.json",
        output_file="train_output1.json",
        num_answers=5,
        max_workers=6,
        resume=True,
        request_interval=1.5
    )
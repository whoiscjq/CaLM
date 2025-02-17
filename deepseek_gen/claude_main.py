import json
import multiprocessing
from functools import partial
import os
import logging
from tqdm import tqdm
import argparse
import requests

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    response = requests.request("POST", url, headers=headers, data=payload, timeout=None)

    # 解析 JSON 数据为 Python 字典
    try:
        data = response.json()
    except:
        print(response)
        return "json_fail"

    # 获取 content 字段的值
    if data.get("choices"):
        return data['choices'][0]['message']['content']
    else:
        print(data)
        return "data_fail"

def process_question(question, num_responses=10):
    responses = [get_response(question) for _ in range(num_responses)]
    return {"question": question, "responses": responses}

def save_checkpoint(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f)
    logging.info(f"Checkpoint saved: {filename}")

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        logging.info(f"Checkpoint loaded: {filename}")
        return data
    logging.info("No checkpoint found, starting from scratch")
    return []

def main(input_file, output_file, num_processes):
    # 读取数据集
    with open(input_file, "r") as f:
        dataset = json.load(f)
    
    questions = [item["question"] for item in dataset]
    total_questions = len(questions)
    
    logging.info(f"Total questions in dataset: {total_questions}")
    
    # 加载断点
    checkpoint_file = f"{output_file}.checkpoint"
    results = load_checkpoint(checkpoint_file)
    processed_questions = set(item["question"] for item in results)
    
    # 过滤掉已经处理过的问题
    questions_to_process = [q for q in questions if q not in processed_questions]
    
    logging.info(f"Questions to process: {len(questions_to_process)}")
    
    logging.info(f"Using {num_processes} processes")
    
    # 使用进程池并行处理问题
    with multiprocessing.Pool(num_processes) as pool:
        # 使用tqdm创建进度条
        pbar = tqdm(total=len(questions_to_process), desc="Processing questions")
        
        for result in pool.imap_unordered(process_question, questions_to_process):
            results.append(result)
            pbar.update(1)
            
            # 每处理10个问题保存一次断点
            if len(results) % 10 == 0:
                save_checkpoint(results, checkpoint_file)
        
        pbar.close()
    
    # 保存最终结果
    save_checkpoint(results, output_file)
    logging.info("Processing completed")

if __name__ == "__main__":
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
    
    
    
    parser = argparse.ArgumentParser(description="Process questions using LLM")
    parser.add_argument("--input_file",type=str,default="claude_dataset/train.json", help="Input JSON file containing questions")
    parser.add_argument("--output_file",type=str,default="claude_dataset/output.json",help="Output JSON file for results")
    parser.add_argument("-p", "--processes", type=int, default=multiprocessing.cpu_count(),
                        help="Number of processes to use (default: number of CPU cores)")
    
    args = parser.parse_args()
    
    #gen_input_file(args.input_file,"claude_dataset/tmp.json")
    main(args.input_file, args.output_file, args.processes)


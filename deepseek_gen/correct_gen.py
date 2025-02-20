import jsonlines
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import json
import requests

# 多API密钥配置（文献7）
API_KEYS = ["7d4fe6b6-590a-4563-8fe0-d3dd7acf11a1"]
current_key_idx = 0
key_lock = threading.Lock()



client = OpenAI(
    api_key = API_KEYS[0],
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)


class APIProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.processed_ids = self._load_processed_ids()
        self.lock = threading.Lock()
        print(self.processed_ids)
        
    def _load_processed_ids(self):
        try:
            #read from jsonl file
            data = []
            with jsonlines.open("answer_check/log.jsonl", 'r') as reader:
                for obj in reader:
                    data.append(obj)
            return set(data)
        except FileNotFoundError:
            return set()

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_response(self, item):
        Baseurl = "https://api.claudeshop.top"
        Skey = "sk-K2dKtLK3O5LlkYr1vcEQ9Ab7SSONSmgB8fCbIJriXm7Of8Vz"
        answer = item["answer"][-30:]
        correct_answer = item["steps"]
        payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages":[{"role":"system",
                    "content":"""I want you to act as a professional answer judge. You will be given a solution and a ground truth answer. Your task is to judge whether the final answer of solution is correct based on the ground truth answer, and you need to check whether the solution give the right final numerical answer. 
                                Moreover, slight numerical differences are allowed. Your answer should be a single word: correct or incorrect.
                                For example, 
                                solution: XXXXX {\"ANSWER\": \"Yes\", \"PROB\": \"0.1359\"} \n\n ground_truth: NDE=0.1358>0 so the answer is Yes.\n\n check: correct
                                solution: XXXXX {\"ANSWER\": \"Yes\", \"PROB\": \"0.4594\"} \n\n ground_truth: NDE=0.999>0 so the answer is Yes.\n\n check: incorrect
                                solution: XXXXX {\"ANSWER\": \"No\", \"PROB\": \"-0.4594\"} \n\n ground_truth: NDE=-0.4594<0 so the answer is No.\n\n check: correct
                            """
        },
        {"role": "user", "content":f"solution: {answer}\n\n ground_truth: {correct_answer}\n\n check: "} ],
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
 
    # def _get_response(self, item):
    #     global current_key_idx
    #     with key_lock:  # 密钥轮换（文献6）
    #         api_key = API_KEYS[current_key_idx % len(API_KEYS)]
    #         current_key_idx += 1
    #     question = item["question"]
    #     correct_answer = item["steps"]
    #     return client.chat.completions.create(
    #         model="ep-20250220000118-84hmh",
    #         messages=[{"role":"system","content":"I want you to act as a professional answer judge. You will be given a step-by-step process and a ground truth answer. Your task is to judge whether the step-by-step process leads to the ground truth answer, slight numerical differences are allowed. Your answer should be a single word: yes or no."},{"role": "user", "content":f"process: {question}\nanswer: {correct_answer}"}],
    #         timeout=400  # 文献5建议的超时设置
    #     )

    def _save_result(self, item):
        with self.lock:
            # 原子化写入（文献4）
            with jsonlines.open(self.output_file, mode='a') as writer:
                writer.write(item)
            # 记录处理进度
            with open("answer_check/log.jsonl", 'a') as f:
                f.write(f"{item['global_id']}\n")

    def process_item(self, item):
        if item["global_id"] in self.processed_ids:
            print(f"已处理[{item['global_id']}]by keeping")
            return 
        # if item.get("answer"):
        #     self._save_result(item)
        #     print(f"已处理[{item['global_id']}]by copy")
        #     return
            
        try:
            response = self._get_response(item)
            item["check"] = response
            print(f"处理成功[{item['global_id']}]")
            self._save_result(item)
        except Exception as e:
            print(f"处理失败[{item['global_id']}]: {str(e)}")
            raise

    def run(self, max_workers=5):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with jsonlines.open(self.input_file) as reader:
                futures = []
                for item in reader:
                    if item["global_id"] not in self.processed_ids:
                        futures.append(executor.submit(self.process_item, item))
                
                for future in futures:
                    future.result()  # 触发异常传播

if __name__ == "__main__":
    processor = APIProcessor(
        input_file="answer_check/deepseek_NDE_output_partial.jsonl",
        output_file="answer_check/deepseek_NDE_check_output.jsonl"
    )
    processor.run(max_workers=8)
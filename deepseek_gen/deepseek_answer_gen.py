import jsonlines
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import os

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
            with jsonlines.open("deepseek_responses/deepseek_NDE_output_log.jsonl", 'r') as reader:
                for obj in reader:
                    data.append(obj)
            return set(data)
        except FileNotFoundError:
            return set()

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_response(self, question):
        global current_key_idx
        with key_lock:  # 密钥轮换（文献6）
            api_key = API_KEYS[current_key_idx % len(API_KEYS)]
            current_key_idx += 1
        
        return client.chat.completions.create(
            model="ep-20250220000118-84hmh",
            messages=[{"role":"system","content":"You are an expert in the field of causal inference and you need to answer the following questions."},{"role": "user", "content": question}],
            timeout=400  # 文献5建议的超时设置
        )

    def _save_result(self, item):
        with self.lock:
            # 原子化写入（文献4）
            with jsonlines.open(self.output_file, mode='a') as writer:
                writer.write(item)
            # 记录处理进度
            with open("deepseek_responses/deepseek_NDE_output_log.jsonl", 'a') as f:
                f.write(f"{item['global_id']}\n")

    def process_item(self, item):
        if item["global_id"] in self.processed_ids:
            print(f"已处理[{item['global_id']}]by keeping")
            return 
        if item.get("answer"):
            self._save_result(item)
            print(f"已处理[{item['global_id']}]by copy")
            return
            
        try:
            response = self._get_response(item["question"])
            item["answer"] = response.choices[0].message.reasoning_content+response.choices[0].message.content
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
        input_file="total_train_dup_5.jsonl",
        output_file="deepseek_responses/deepseek_NDE_output.jsonl"
    )
    processor.run(max_workers=8)
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            data.append(json.loads(line))
        return data


class Processor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.processed_ids = self._load_processed_ids()
        
    def _load_processed_ids(self):
        # try:
        #     with open('processed.log', 'r') as f:
        #         return set(f.read().splitlines())
        # except FileNotFoundError:
        #     return set()
        return set()

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_response(self, item):
        input_text = item["steps"]
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", 
                 "content": "Convert the input text into a JSON object by following these rules:\n\n1. Extract the numerical value after `NDE=` and before the comparison operator (`>` or `<`).\n2. Determine the `ANSWER` based on the comparison:\n   - If `>0`, set `ANSWER` to \"Yes\".\n   - If `<0`, set `ANSWER` to \"No\".\n3. Format the output as: {\"ANSWER\": \"Yes/No\", \"PROB\": \"[extracted_value]\"}\n\nExamples:\n- Input: \"NDE=0.3171>0 so the answer is Yes.\" → Output: {\"ANSWER\": \"Yes\", \"PROB\": \"0.3171\"}\n- Input: \"NDE=-0.3424<0 so the answer is No.\" → Output: {\"ANSWER\": \"No\", \"PROB\": \"-0.3424\"}\n\nRequirements:\n- Preserve the exact numerical value (including sign and decimals).\n- Use double quotes for JSON keys/values.\n- Do not include additional fields or explanations."},
                {"role": "user", "content": input_text}
            ]
        })
        
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer sk-K2dKtLK3O5LlkYr1vcEQ9Ab7SSONSmgB8fCbIJriXm7Of8Vz',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            "https://api.claudeshop.top/v1/chat/completions",
            headers=headers,
            data=payload,
            timeout=30
        )
        response.raise_for_status()
        return item, response

    def parse_response(self, item, response):
        try:
            json_data = response.json()
            if json_data.get("choices"):
                item["gt_answer"] = json_data['choices'][0]['message']['content']
                return item
        except Exception as e:
            print(f"解析失败: {item.get('global_id')}, 错误: {str(e)}")
        return None

    def process_item(self, item):
        if str(item.get("global_id")) in self.processed_ids:
            return None
            
        try:
            item, response = self.get_response(item)
            return self.parse_response(item, response)
        except Exception as e:
            print(f"处理失败: {item.get('global_id')}, 错误: {str(e)}")
            return item  # 返回原item用于重试

    def run(self):
        # with open(self.input_path) as f:
        #     data = json.load(f)
        #read jsonl
        data = read_jsonl(self.input_path)
            
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.process_item, item): item 
                      for item in data if str(item.get("global_id")) not in self.processed_ids}
            
            with open(self.output_path, 'a') as f_out, open('processed.log', 'a') as f_log:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        # 成功处理
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        print(result)
                        #f_log.write(f"{result['global_id']}\n")
                    else:
                        # 失败重试（根据需求可加入重试队列）
                        pass

if __name__ == "__main__":
    processor = Processor(
        input_path="answer_check/deepseek_NDE_check_output.jsonl",
        output_path="answer_check/deepseek_NDE_check_gt_output.jsonl"
    )
    processor.run()
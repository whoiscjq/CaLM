import json
import tqdm
import os
import random
import openai
from datetime import datetime
import argparse
import time
import requests
    

def make_requests(
        engine, prompts, max_tokens, temperature,
        frequency_penalty, presence_penalty, 
        n, retries=3, api_key=None, organization=None
    ):
    response = None
    target_length = max_tokens
    retry_cnt = 0
    backoff_time = 30
    print("hello2")
    while retry_cnt <= retries:
        try:
            Baseurl = "https://api.claudeshop.top"
            Skey = api_key
            
            payload = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompts}],
                "max_tokens":target_length,
                "temperature":temperature,
                "frequency_penalty":frequency_penalty,
                "presence_penalty":presence_penalty,
                "n":n,
                })
            url = Baseurl + "/v1/chat/completions"
            headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {Skey}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            break
        # except: openai.error.OpenAIError as e:
        #     print(f"OpenAIError: {e}.")
        #     if "Please reduce your prompt" in str(e):
        #         target_length = int(target_length * 0.8)
        #         print(f"Reducing target length to {target_length}, retrying...")
        #     else:
        #         print(f"Retrying in {backoff_time} seconds...")
        #         time.sleep(backoff_time)
        #         backoff_time *= 1.5
        except:
            print(f"error time:{retry_cnt}")
            retry_cnt += 1
    
    return response.json()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo",
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=1024,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="The `n` parameter of GPT3. The number of responses to generate."
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    return parser.parse_args()

    
if __name__ == "__main__":
    random.seed(123)
    args = parse_args()

    dirname = os.path.dirname(args.output_file)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        all_prompts = [json.loads(line) for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts))):
            prompt = all_prompts[i]["prompt"]
            if prompt in existing_responses:
                fout.write(json.dumps(existing_responses[prompt]) + "\n")
            else:
                response = make_requests(
                    engine=args.engine,
                    prompts=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    n=args.n,
                    api_key=args.api_key
                )
                # print(all_prompts[i]['node'], all_prompts[i]['edge'], response["choices"][0]["message"]["content"])
                # print(all_prompts[i]['node_name'], response["choices"][0]["message"]["content"])
                for key in all_prompts[i]:
                    if key == "prompt": continue
                    #print(all_prompts[i][key])
                #print(response["choices"][0]["message"]["content"])
                print("******************************")
                all_prompts[i]["response"] = response
                fout.write(json.dumps(all_prompts[i], ensure_ascii=False) + "\n")

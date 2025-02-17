    ```bash
    python gen_prompt.py --example_num 3 --graph_file data/graphs/4_nodes.txt --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_prompt.jsonl --repeat_num 2
    python gpt3_api.py --api_key=sk-gQUCqamY623o1vzuYJtXwHVDSlKyD2vWMs954WoYI7JmCwWB --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_prompt.jsonl  --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_result.jsonl
    ```
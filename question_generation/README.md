# question_generation

1. `main.py` 给定的因果图（节点具有实际含义），自动生成一些问题和对应的答案。
    ```bash
    python main.py --types judge --output_file data/questions/output_judge.json  # 是非判断题
    python main.py --types general --output_file data/questions/output_general.json  # 简答题
    python main.py --types compute --output_file data/questions/output_compute.json  # 基于数据计算概率回答问题
    python main.py --types judge general compute --output_file data/questions/output_all.json  # 生成所有类型的问题
    ```
2. `gen_graph.py` 给定节点数，生成所有仅包含一个连通域的、结构不重复的有向无环图。若指定`-g`为大于0的数，则为每种节点数随机采样所指定个数的DAG。若指定`-g`为小于等于0的数，则遍历所有可能的DAG。
    ```bash
    python gen_graph.py -n 7 50 --o result.txt -g 500  # 生成节点数在[7,50]内的DAG，每种节点数500个graph
    python gen_graph.py -n 2 6 --o result.txt  # 生成节点数在[2,6]内的DAG，遍历每种节点所有DAG
    python gen_graph.py -n 3 --o result.txt  # 生成节点数为3的DAG，遍历每种节点所有DAG
    ```
3. `gen_prompt.py` 节点赋值prompt，然后调用`gpt3_api.py`进行节点赋值。
    ```bash
    python gen_prompt.py --example_num 3 --graph_file data/graphs/4_nodes.txt --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_prompt.jsonl --repeat_num 2
    python gpt3_api.py --api_key=sk-gQUCqamY623o1vzuYJtXwHVDSlKyD2vWMs954WoYI7JmCwWB --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_prompt.jsonl  --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_result.jsonl
    ```
5. `gen_translation_prompt.py` 生成翻译prompt，然后调用`gpt3_api.py`进行翻译及生成表述节点的形容词
    ```bash
    python gen_translate_prompts.py --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_result.jsonl --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_translate_prompt.jsonl
    python gpt3_api.py --api_key=xxxx --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_translate_prompt.jsonl  --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_translate_result.jsonl
    python update_translate_dict.py --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_translate_result.jsonl  # 更新字典
    ```
6. `gen_relation_prompt.py` 生成边的符号（正相关/负相关）赋值prompt，然后调用`gpt3_api.py`进行边的符号赋值。
    ```bash
    python gen_relation_prompts.py --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_result.jsonl  --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_relation_prompt.jsonl
    python gpt3_api.py --api_key=sk-gQUCqamY623o1vzuYJtXwHVDSlKyD2vWMs954WoYI7JmCwWB --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_relation_prompt.jsonl  --output_file data/gpts/prompt_and_response/4_nodes_repeat_2_relation_result.jsonl
    python update_relation_prompt.py --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_relation_result.jsonl # 更新字典
    ```
7. `preprocess_graph.py` 预处理整理图、节点含义、边符号到一个文件
    ```bash
    python preprocess_graph.py --input_file data/gpts/prompt_and_response/4_nodes_repeat_2_result.jsonl --output_file data/gpts/anno_test_data/4_nodes.json
    ```

8. `gen_causal_questions.py` 生成因果问答数据
    ```bash
    python gen_causal_questions.py --input_file data/gpts/anno_test_data/4_nodes.json --output_prefix data/gpts/final_test_data/4_nodes --node_num 4
    ```


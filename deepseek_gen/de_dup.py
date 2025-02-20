import jsonlines

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries representing the JSONL file.
    """
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data

def write_jsonl(file_path, data):
    """
    Writes a list of dictionaries to a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.
        data (list): A list of dictionaries to be written to the JSONL file.
    """
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(data)

def de_dup(input_file, output_file,output_log):
    #dup by global_id
    data = read_jsonl(input_file)
    ids=set()
    clean_data=[]
    for item in data:
        id=item["global_id"]
        if id not in ids:
            ids.add(id)
            clean_data.append(item)
    write_jsonl(output_file, clean_data)
    write_jsonl(output_log, list(ids))

de_dup(input_file="/mnt/workspace/o1_like/CaLM/deepseek_gen/train_ark_output.jsonl",output_file="deepseek_responses/deepseek_NDE_output.jsonl",output_log="deepseek_responses/deepseek_NDE_output_log.jsonl")
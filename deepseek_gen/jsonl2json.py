import json

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def save_as_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def jsonl2json(input_file, output_file):
    data=read_jsonl(input_file)
    save_as_json(data, output_file)

if __name__ == "__main__":
    input_file = "dataset/NDE/test_gt.jsonl"
    output_file = "dataset/NDE/test_gt.json"
    jsonl2json(input_file, output_file)
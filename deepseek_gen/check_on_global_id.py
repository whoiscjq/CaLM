import jsonlines
import os

#read jsonl file
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

#write jsonl file
def write_jsonl(file_path, data):
    """
    Writes a list of dictionaries to a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.
        data (list): A list of dictionaries to be written to the JSONL file.
    """
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(data)

data = read_jsonl("")
ids=set()
clean_data=[]
for item in data:
    id=item["global_id"]
    if id not in ids:
        ids.add(id)
        clean_data.append(item)
write_jsonl("", clean_data)
write_jsonl("", list(ids))
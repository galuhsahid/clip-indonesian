"""
Check whether the generated JSON Lines file
is valid or not.

Sample usage:
python check_jsonl.py <jsonl file to be checked>
python check_jsonl.py flickr8k_dataset_train.json
"""

import json
import sys

file_name = sys.argv[1]

with open(file_name, 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    try:
        result = json.loads(json_str)

        if not isinstance(result, dict):
            print(json_str)
    except Exception as e:
        print(json_str)
        print(e)

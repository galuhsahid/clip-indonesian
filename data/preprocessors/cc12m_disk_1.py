"""
This script:
- converts the original CC12M dataset file to jsonl files
- creates image paths based on the given image folder
- split the dataset into training and validation splits (90%:10% split)

Sample usage:
python preprocessors/cc12m_disk_1.py <CC12M tsv file> <CC12M images folder> <output file name (without extension)>
python preprocessors/cc12m_disk_1.py datasets/cc12m/cc12m_translate_final.tsv disk1/datasets/cc12m/images datasets/cc12m/cc12m_dataset_disk1

The script will output two files: cc12m_dataset_disk1_train.json and cc12m_dataset_disk1_val.json
"""

import pandas as pd
import os.path
import sys
import json
import logging
import contexttimer
import numpy as np

# Setup
logging.basicConfig(filename='download.log', filemode='w', level=logging.INFO)

if len(sys.argv) != 4:
    print("Provide .tsv file name, images dir, output file name. e.g. python preprocessors/cc12m_disk_1.py datasets/cc12m/cc12m_translate_final.tsv disk1/datasets/cc12m/images datasets/cc12m/cc12m_dataset_disk1")
    exit(1)

annotation_file = sys.argv[1]
images_dir = sys.argv[2]
output_file = sys.argv[3]

logging.info("Processing cc12m dataset")

with contexttimer.Timer(prefix="Loading from tsv"):
    df = pd.read_csv(annotation_file, delimiter='\t')

lines = []

df = df[["caption", "url"]]

df = df.replace('', np.nan)
df = df.dropna()

print(f"Loaded {len(df)} images.")

for index, caption_reference_description, image_url in df.itertuples():
    index+=1
    base_url = os.path.basename(image_url)  # extract base url
    stem, ext = os.path.splitext(base_url)  # split into stem and extension
    filename = f'{index:08d}---{stem}.jpg'

    full_image_path =  images_dir+"/"+filename

    if os.path.isfile(full_image_path):
        lines.append(json.dumps({"image_path": full_image_path, "captions": [caption_reference_description]}))
    else:
        #print(f"{full_image_path} doesn't exist")
        logging.error(full_image_path)

train_lines = lines[:-150_001]
valid_lines = lines[-150_001:]

with open(output_file+"_train.json", "w") as f:
    f.write("\n".join(train_lines))

with open(output_file+"_val.json", "w") as f:
    f.write("\n".join(valid_lines))

logging.info(f"Processing cc12m dataset done. {len(lines)} images processed.")

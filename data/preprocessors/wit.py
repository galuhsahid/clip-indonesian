"""
This script:
- converts the original WiT dataset file to jsonl files
- creates image paths based on the given image folder
- split the dataset into training and validation splits (90%:10% split)

Sample usage:
python preprocessors/wit.py <wit tsv file> <wit images folder> <output file name (without extension)>
python preprocessors/wit.py datasets/wit/wit_id_filtered.tsv datasets/wit/images datasets/wit/wit_dataset

The script will output two files: wit_dataset_train.json and wit_dataset_val.json
"""

import pandas as pd
import os.path
import sys
import json
import logging
import contexttimer
import numpy as np

if len(sys.argv) != 4:
    print("Provide .tsv file name, images dir, output file name. e.g. python coco.py coco_captions_train2017.json /mnt/disks/data-1/flickr8k/coco_train.json coco_dataset_train.json")
    exit(1)

annotation_file = sys.argv[1]
images_dir = sys.argv[2]
output_file = sys.argv[3]

logging.info("Processing WIT dataset")

with contexttimer.Timer(prefix="Loading from tsv"):
    df = pd.read_csv(annotation_file, delimiter='\t')

images_dict = {}

lines = []

df = df[["caption_reference_description", "image_url"]]

df = df.replace('', np.nan)
df = df.dropna()

for index, caption_reference_description, image_url in df.itertuples():
    base_url = os.path.basename(image_url)  # extract base url
    stem, ext = os.path.splitext(base_url)  # split into stem and extension
    filename = f'{stem}.jpg'

    full_image_path =  images_dir+"/"+filename

    if os.path.isfile(full_image_path):
        lines.append(json.dumps({"image_path": full_image_path, "captions": [caption_reference_description]}))
    else:
        print(f"{full_image_path} doesn't exist")

train_lines = lines[:-9_001]
valid_lines = lines[-9_001:]

with open(output_file+"_train.json", "w") as f:
    f.write("\n".join(train_lines))

with open(output_file+"_val.json", "w") as f:
    f.write("\n".join(valid_lines))

logging.info(f"Processing WiT dataset done. {len(lines)} images processed.")

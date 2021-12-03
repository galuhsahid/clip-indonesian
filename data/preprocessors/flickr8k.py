"""
This script:
- converts the original flickr8k dataset file to jsonl files
- creates image paths based on the given image folder
- split the dataset into training and validation splits (90%:10% split)

Sample usage:
python preprocessors/flickr8k.py <flickr8k json file> <flickr8k images folder> <output file name (without extension)>
python preprocessors/flickr8k.py datasets/flickr8k/flickr8k_train.json datasets/flickr8k/images datasets/flickr8k/flickr8k_dataset

The script will output two files: flickr8k_dataset_train.json and flickr8k_dataset_val.json
"""

import json
import logging
import sys
import os.path

if len(sys.argv) != 4:
    print("Provide .tsv file name, images dir, output file name. e.g. python preprocessors/flickr8k.py datasets/flickr8k/flickr8k_train.json datasets/flickr8k/images datasets/flickr8k/flickr8k_dataset")
    exit(1)

annotation_file = sys.argv[1]
images_dir = sys.argv[2]
output_file = sys.argv[3]

logging.info("Processing Flickr 8k dataset")

with open(annotation_file, "r") as f:
    annotations = json.load(f)

lines = []
for image_path, captions in annotations.items():
    edited_captions = []
    for caption in captions:
        if len(caption) > 0:
            edited_captions.append(caption.replace("<start> ", "").replace(" <end>", ""))
    full_image_path =  images_dir+"/"+image_path
    if os.path.isfile(full_image_path):
        if len(edited_captions) > 0:
            lines.append(json.dumps({"image_path": full_image_path, "captions": edited_captions}))
    else:
        print(f"{full_image_path} doesn't exist")

# Train and validation split
train_lines = lines[:-801]
valid_lines = lines[-801:]

with open(output_file+"_train.json", "w") as f:
    f.write("\n".join(train_lines))

with open(output_file+"_val.json", "w") as f:
    f.write("\n".join(valid_lines))

logging.info(f"Processing Flickr 8k dataset done. {len(lines)} images processed.")

"""
This script:
- converts the original COCO dataset file to jsonl files
- creates image paths based on the given image folder
- split the dataset into training and validation splits (90%:10% split)

Sample usage:
python preprocessors/coco.py <coco json file> <coco images folder> <output file name (without extension)>
python preprocessors/coco.py datasets/coco/coco_captions_train2017.json datasets/coco/images datasets/coco/coco_dataset

The script will output two files: coco_dataset_train.json and coco_dataset_val.json
"""

import json
import collections
import logging
import sys

if len(sys.argv) != 4:
    print("Provide .tsv file name, images dir, output file name. e.g. python preprocessors/coco.py datasets/coco/coco_captions_train2017.json datasets/coco/images datasets/coco/coco_dataset")
    exit(1)

annotation_file = sys.argv[1]
images_dir = sys.argv[2]
output_file = sys.argv[3]

logging.info("Processing COCO dataset")

with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]

image_path_to_caption = collections.defaultdict(list)
for element in annotations:
    caption = f"{element['caption'].lower().rstrip('.')}"
    image_path = images_dir + "/%012d.jpg" % (element["image_id"])
    image_path_to_caption[image_path].append(caption)

lines = []
for image_path, captions in image_path_to_caption.items():
    lines.append(json.dumps({"image_path": image_path, "captions": captions}))

# Train and validation split
train_lines = lines[:-10_001]
valid_lines = lines[-10_001:]

with open(output_file+"_train.json", "w") as f:
    f.write("\n".join(train_lines))

with open(output_file+"_val.json", "w") as f:
    f.write("\n".join(valid_lines))

logging.info(f"Processing COCO dataset done. {len(lines)} images processed.")

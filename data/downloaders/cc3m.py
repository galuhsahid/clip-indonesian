"""
This script was adapted from Luke Melas-Kyriazi's code with changes
made for the CC3M Dataset.

Read more and download the CC3M dataset from: https://github.com/google-research-datasets/conceptual-captions
CC3M with captions translated to Indonesian using Marian: https://github.com/acul3/translated-dataset#cc3m

Sample usage:
python downloaders/cc3m.py <tsv file> <output folder>
python downloaders/cc3m.py datasets/cc3m/cc3m.tsv datasets/cc3m/images
"""

import sys
import os
from datetime import datetime
import pandas as pd
import contexttimer
import requests
from PIL import Image
from torchvision.transforms import functional as TF
from multiprocessing import Pool
from tqdm import tqdm
import logging
import sys

# Setup
logging.basicConfig(filename='download.log', filemode='w', level=logging.INFO)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

if len(sys.argv) != 3:
    print("Provide .tsv file name & output directory. e.g. python downloader.py Train-GCC-training.tsv training")
    exit(1)

# Load data
print(f'Starting to load at {datetime.now().isoformat(timespec="minutes")}')
with contexttimer.Timer(prefix="Loading from tsv"):
    df = pd.read_csv(sys.argv[1], delimiter='\t', header=None)

url_to_idx_map = {url: index for index, caption, url in df.itertuples()}
print(f'Loaded {len(url_to_idx_map)} urls')

base_dir = os.path.join(os.getcwd(), sys.argv[2])

def process(item):
    url, image_id = item
    try:
        base_url = os.path.basename(url)  # extract base url
        stem, ext = os.path.splitext(base_url)  # split into stem and extension
        filename = f'{image_id:08d}---{stem}.jpg'  # create filename
        filepath = os.path.join(base_dir, filename)  # concat to get filepath
        if not os.path.isfile(filepath):
            req = requests.get(url, stream=True, timeout=1, verify=False).raw
            image = Image.open(req).convert('RGB')
            if min(image.size) > 512:
                image = TF.resize(image, size=512, interpolation=Image.LANCZOS)
            image.save(filepath)  # save PIL image
    except Exception as e:
        logging.info(" ".join(repr(e).splitlines()))
        logging.error(url)

list_of_items = list(url_to_idx_map.items())
print(len(list_of_items))

with Pool(128) as p:
    r = list(tqdm(p.imap(process, list_of_items), total=len(list_of_items)))
    print('DONE')

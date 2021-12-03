"""
This script was adapted from Luke Melas-Kyriazi's code with changes
made for the Wikipedia-based Image Text (WiT) Dataset.

Before downloading the images we also filter the data by language (line 46).

Read more and download the WiT dataset from: https://github.com/google-research-datasets/wit

Sample usage:
python downloaders/wit.py <tsv file> <output folder>
python downloaders/wit.py datasets/wit/wit.tsv datasets/wit/images
"""

import sys
import os
from datetime import datetime
import pandas as pd
import contexttimer
from urllib.request import urlopen
import requests
from PIL import Image
from torchvision.transforms import functional as TF
from multiprocessing import Pool
from tqdm import tqdm
import logging
import sys

headers = {
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    "X-Forwarded-For": "64.18.15.200",
}

# Setup
logging.basicConfig(filename='download.log', filemode='w', level=logging.INFO)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

if len(sys.argv) != 3:
    print("Provide .tsv file name & output directory. e.g. python wit.py wit.tsv images")
    exit(1)

# Load data
print(f'Starting to load at {datetime.now().isoformat(timespec="minutes")}')

with contexttimer.Timer(prefix="Loading from tsv"):
    df = pd.read_csv(sys.argv[1], delimiter='\t')
    df = df[df["language"] == "id"] # filter by language

    df = df[["caption_reference_description", "image_url"]]
    df.columns = ["caption", "url"]

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
            req = requests.get(url, stream=True, timeout=10, allow_redirects=True, verify=False, headers=headers).raw
            image = Image.open(req).convert('RGB')
            if min(image.size) > 512:
                image = TF.resize(image, size=512, interpolation=Image.LANCZOS)
            image.save(filepath)  # save PIL image
    except Exception as e:
        logging.info(" ".join(repr(e).splitlines()))
        logging.error(url)

if __name__ == '__main__':
    list_of_items = list(url_to_idx_map.items())
    print(len(list_of_items))

    # Multiprocessing enables a faster downloading process, however approximately ~20%
    # of images will be lost. This might not be a problem for CC3M and CC12M that are
    # large datasets, but it's a problem for WiT data that only have ~100k of images-caption
    # pairs for Indonesian data. Thus the decision to download the images without multiprocessing
    # in order to preserve all images as many as possible.
    for url, image_id in tqdm(list_of_items):
        item = (url, image_id)
        process(item)

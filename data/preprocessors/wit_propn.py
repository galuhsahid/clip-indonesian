"""
Filter out image-text pairs with captions that have proper nouns
for 80% of the caption. Only applies to the WiT dataset.

This means that captions that consist of mostly proper nouns, such as
"Budi Budiman" will be filtered out and not used.

Source for the POS tagger: https://yudiwbs.wordpress.com/2018/02/20/pos-tagger-bahasa-indonesia-dengan-pytho/

Sample usage:
python preprocessors/wit_propn.py <input tsv file> <output tsv file> <minimum threshold of propn percentage (0 to 1)>
python preprocessors/wit_propn.py datasets/wit/wit.tsv datasets/wit/wit_filtered.tsv 0.8
"""

import sys
from datetime import datetime
import pandas as pd
import contexttimer
from torchvision.transforms import functional as TF
from multiprocessing import Pool
from tqdm import tqdm
import sys
import numpy as np
from nltk.tag import CRFTagger

# Setup CRFTagger
ct = CRFTagger()
ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')

# Load data
print(f'Starting to load at {datetime.now().isoformat(timespec="minutes")}')

with contexttimer.Timer(prefix="Loading from tsv"):
    df = pd.read_csv(sys.argv[1], delimiter='\t')
    df = df[["caption_reference_description", "image_url"]]

def drop_propn(text):
    try:
        if len(text)==0:
            return True
        text = text.split()
        result = ct.tag_sents([text])
        nnp_cnt = 0
        total = len(result[0])

        for x in result[0]:
            if x[1] == "NNP":
                nnp_cnt += 1      
        
        if (nnp_cnt/total) >= sys.argv[3]:
            return True
        return False
    except Exception as e:
        print(e)
        return True

df["to_drop"] = df["caption_reference_description"].apply(drop_propn)

df = df[df["to_drop"]==False]
df = df.drop("to_drop",axis=1)

df.to_csv(sys.argv[2], sep='\t')

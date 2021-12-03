# Data

## Datasets

| Name | Count (Train)* | Count (Validation)* | Original Dataset Link | Translated Annotations Link
| --- | ----------- | ----------- | ----------- | ----------- |
| CC12M | 9,480,140 | 650,000 | [Link](https://github.com/google-research-datasets/conceptual-12m) | [Link](https://github.com/acul3/translated-dataset#cc12m)
| CC3M | 2,520,816 | 300,000 | [Link](https://ai.google.com/research/ConceptualCaptions/) | [Link](https://github.com/acul3/translated-dataset#cc3m)
| COCO 2017 | 108,285 | 10,000 | [Link](https://cocodataset.org/) | [Link](https://github.com/acul3/translated-dataset#coco-2017-train)
| Flickr8k | 5,670 | 800 | [Link](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) | [Link](https://drive.google.com/uc?id=1myBppMVzHuHluiSaWRykA-rBMPrgZELd)
| WiT | 89,610 | 9,000 | [Link](https://github.com/google-research-datasets/wit) | Dataset is already in Indonesian (filter by `lang` = `id`)
| Total | 12,204,521 | 969,800 |  | 

\*) excludes broken images, SVGs, and images that cannot be downloaded. For WiT, also excludes image-text pairs with captions that have 80% of proper nouns.

## Scripts

### Downloaders
The images for CC3M, CC12M, and WiT need to be downloaded separately from the annotations. The scripts for downloading these images are located in `/downloaders`. Please note that images may go missing at anytime and thus the number of downloaded images may vary from time to time.

### Preprocessors
The Hybrid CLIP script accepts JSON lines (jsonl) files as input. The scripts in the `/preprocessors` folder convert JSON or .tsv files (depending on the dataset) into JSON lines files. Each dataset will have a separate `train` and `val` dataset.

### Others

The files that are the outputs of the preprocessors scripts will then be concatenated into one large train JSON lines file and one large validation JSON lines file as inputs for the Hybrid CLIP script. To accomplish this we can use scripting languages such as bash or awk - though awk is recommended because it ensures every line to have a new line during the concatenation process.

Example:

```
awk 1 cc12m_dataset_disk1_train.json cc12m_dataset_disk2_train.json cc3m_dataset_train.json coco_dataset_train.json  flickr8k_dataset_train.json wit_dataset_train.json > train_dataset_v4.json
```

```
awk 1 cc12m_dataset_disk1_val.json cc12m_dataset_disk2_val.json cc3m_dataset_val.json coco_dataset_val.json flickr8k_dataset_val.json wit_dataset_val.json > val_dataset_v4.json
```

We also have a `check_jsonl.py` script to help check whether the generated final train and validation JSON lines files are valid or not.
# CLIP-Indonesian

CLIP ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) is a multimodal model that can connect images and text by training a vision encoder and a text encoder jointly to project the representation of images and the corresponding text into the same embedding space. The expected outcome is the text embeddings and image embeddings are located near each other.

This repository hosts the code for CLIP-Indonesian, which is a CLIP multimodal model trained on Indonesian data.

For the image encoder, we use [VIT](https://huggingface.co/models?filter=vit), more specifically `openai/clip-vit-base-patch32`. Meanwhile, for the text encoder, we experimented with two models: IndoBERT Large (`indobenchmark/indobert-base-p2`) and Indonesian RoBERTa Base (`flax-community/indonesian-roberta-base`).

Most of the CLIP script is based on [HybridCLIP](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/hybrid_clip) and [clip-italian](https://arxiv.org/abs/2108.08688).

Still a work in progress so may not give the best result (yet) :)

clip-indonesian was presented in PyCon ID 2021. You can view the slide deck [here](https://bit.ly/pycon-clip-indonesian).

# Dataset

More details about the dataset used can be found [here](/data).

# Results

The results of the training can be accessed [here](https://wandb.ai/galuh/clip-indonesian).

# Demo

- [Zero-shot image classification](https://colab.research.google.com/drive/19p4f7eLnKp8Dxp0tjiEiWMoMv3jpXz48?usp=sharing)
- [Image search on Unsplash25k dataset](https://colab.research.google.com/drive/1v56LQLpNB8z0PwMQ9uESLRpGM-hO9aRf?usp=sharing)

# Links

+ [GitHub Repository](https://github.com/galuhsahid/clip-indonesian)
+ [Model on HuggingFace](https://huggingface.co/Galuh/clip-indonesian)

# References

Bianchi, F., Attanasio, G., Pisoni, R., Terragni, S., Sarti, G., Lakshmi, S. (2021). [Contrastive Language-Image Pre-training for the Italian Language](https://arxiv.org/abs/2108.08688) arXiv preprint arXiv:2108.08688.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). [Learning Transferable Visual Models From Natural Language Supervision.](https://arxiv.org/abs/2103.00020) ICML.

Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., ... & Purwarianti, A. (2020). [IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding](https://arxiv.org/pdf/2009.05387.pdf). arXiv preprint arXiv:2009.05387.

[Hybrid CLIP](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/hybrid_clip) by the HuggingFace team

[Indonesian Roberta Base](https://huggingface.co/flax-community/indonesian-roberta-base) by Wilson Wongso, Steven Limcorn, Samsul Rahmadani, and Chew Kok Wah

[Indonesian Translated Datasets](https://github.com/acul3/translated-dataset) by Samsul Rahmadani

# Acknowledgment
All training was done on a TPUv3-8 VM sponsored by [TPU Research Cloud](https://sites.research.google/trc/about/).

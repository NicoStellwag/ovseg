# Language-driven Semantic Segmentation (LSeg)


## Usage
### Installation
1. install dependencies for Mask3D
torchmetrics==0.11.4
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install timm
pip install git+https://github.com/openai/CLIP.git

Download the model for demo and put it under folder `checkpoints` as `checkpoints/demo_e200.ckpt`.
<a href="https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing">download</a>


### Data Preparation
By default, for training, testing and demo, we use [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

```
python prepare_ade20k.py
unzip ../datasets/ADEChallengeData2016.zip
```


## Acknowledgement
The repo contains a modified PyTorch Implementation of paper [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546). 

ICLR 2022

#### Authors: 
* [Boyi Li](https://sites.google.com/site/boyilics/home)
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)
* [Serge Belongie](https://scholar.google.com/citations?user=ORr4XJYAAAAJ&hl=zh-CN)
* [Vladlen Koltun](http://vladlen.info/)
* [Rene Ranftl](https://scholar.google.at/citations?user=cwKg158AAAAJ&hl=de)

```
@inproceedings{
li2022languagedriven,
title={Language-driven Semantic Segmentation},
author={Boyi Li and Kilian Q Weinberger and Serge Belongie and Vladlen Koltun and Rene Ranftl},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=RriDjddCLN}
}
```

Thanks to the code base from [DPT](https://github.com/isl-org/DPT), [Pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), [CLIP](https://github.com/openai/CLIP), [Pytorch Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [Streamlit](https://streamlit.io/), [Wandb](https://wandb.ai/site)

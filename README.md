## Introduction
This repo attempts to reproduce this amazing work by Kaiming He et al. :
[Mask R-CNN](https://arxiv.org/abs/1703.06870)

## TO-DO
- [] build network structure of 'res_nets_v1_50'

## Step
1. Download pretrained resnet50 model, `wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz` and unzip it
```
tar -xzvf resnet_v1_50_2016_08_28.tar.gz
```
2. Download [COCO](http://mscoco.org/dataset/#download) dataset, place it into `./data`, then run `python download_and_convert_data.py` to build tf-records. It takes a while.

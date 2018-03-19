## Introduction
This repo attempts to reproduce this amazing work by Kaiming He et al. :
[Mask R-CNN](https://arxiv.org/abs/1703.06870)

## TO-DO
- [x] build network structure of 'res_nets_v1_50'
- [x] Download COCO 2017 dataset
- [ ] Use TFRecoder to load data


## Step
1. Download pretrained resnet50 model, `wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz` and unzip it
```
tar -xzvf resnet_v1_50_2016_08_28.tar.gz
```
2. Compile and Install pycocotools
#To compile and install locally run "python setup.py build_ext --inplace"#
#To install library to Python site-packages run "python setup.py build_ext install"#

3. Download [COCO](http://mscoco.org/dataset/#download) dataset, place it into `./data`, then run `python download_and_convert_data.py` to build tf-records. It takes a while.
## Acknowledgment
This project This repo borrows tons of code from
https://github.com/CharlesShang/FastMaskRCNN

## LICENSE
Apache LICENSE 2.0

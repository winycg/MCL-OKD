﻿
# Online Knowledge Distillation for Image Classification

This project provides source code for our Multi-view contrastive learning for online knowledge distillation (MCL-OKD).

## Installation

### Requirements

Ubuntu 16.04 LTS

Python 3.8

CUDA 11.1

PyTorch 1.6.0

Create three folders `./data`, `./result`, and `./checkpoint`,
## Perform experiments on CIFAR-100 dataset
### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder
### Training for baseline
```
python main_cifar_baseline.py --arch densenetd40k12 --gpu 0 
python main_cifar_baseline.py --arch resnet32 --gpu 0 
python main_cifar_baseline.py --arch vgg16 --gpu 0 
python main_cifar_baseline.py --arch resnet110 --gpu 0 
python main_cifar_baseline.py --arch hcgnet_A1 --gpu 0 
```
### Training by DML
```
python main_cifar_dml.py --arch dml_densenetd40k12 --gpu 0
python main_cifar_dml.py --arch dml_resnet32 --gpu 0
python main_cifar_dml.py --arch dml_vgg16 --gpu 0
python main_cifar_dml.py --arch dml_resnet110 --gpu 0
python main_cifar_dml.py --arch dml_hcgnet_A1 --gpu 0
```
### Training by CL-ILR
```
python main_cifar_cl_ilr.py --arch cl_ilr_densenetd40k12 --gpu 0 
python main_cifar_cl_ilr.py --arch cl_ilr_resnet32 --gpu 0 
python main_cifar_cl_ilr.py --arch cl_ilr_vgg16 --gpu 0 
python main_cifar_cl_ilr.py --arch cl_ilr_resnet110 --gpu 0 
python main_cifar_cl_ilr.py --arch cl_ilr_hcgnet_A1 --gpu 0
```

### Training by ONE
```
python main_cifar_one.py --arch one_densenetd40k12 --gpu 0
python main_cifar_one.py --arch one_resnet32 --gpu 0
python main_cifar_one.py --arch one_vgg16 --gpu 0
python main_cifar_one.py --arch one_resnet110 --gpu 0
python main_cifar_one.py --arch one_hcgnet_A1 --gpu 0
```

### Training by OKDDip
```
python main_cifar_okddip.py --arch okddip_densenetd40k12 --gpu 0
python main_cifar_okddip.py --arch okddip_resnet32 --gpu 0
python main_cifar_okddip.py --arch okddip_vgg16 --gpu 0
python main_cifar_okddip.py --arch okddip_resnet110 --gpu 0
python main_cifar_okddip.py --arch okddip_hcgnet_A1 --gpu 0
```

### Training by MCL-OKD
```
python main_cifar_mcl_okd.py --arch mcl_okd_densenetd40k12 --nce_k 256 --gpu 0
python main_cifar_mcl_okd.py --arch mcl_okd_resnet32 --nce_k 256 --gpu 0
python main_cifar_mcl_okd.py --arch mcl_okd_vgg16 --nce_k 16384 --gpu 0
python main_cifar_mcl_okd.py --arch mcl_okd_resnet110 --nce_k 256 --gpu 0
python main_cifar_mcl_okd.py --arch mcl_okd_hcgnet_A1 --nce_k 16384 --gpu 0
```


| Model | FLOPs | Baseline|DML (Ens)|CL-ILR (Ens)|ONE (Ens)|OKDDip (Ens)| MCL-OKD (Ensemble) | 
| - | - | - |
| DenseNet-40-12| 0.07G|29.17|27.34 (26.02)|27.38 (26.19)|29.01 (28.67)|28.75 (27.51)| **26.04 (23.55)** |
| ResNet-32 |0.07G| 28.91|24.92 (22.97)|25.40 (24.03)|25.74 (24.03)|25.76 (23.73)| **24.52 (22.00)** |
| VGG-16 | 0.31G|25.18|24.14 (23.27) | 23.58 (22.96)|25.22 (25.12)|24.86 (24.52)|**23.11 (22.36)** |
| ResNet-110 | 0.17G|23.62|21.51 (19.12) |21.16 (18.66)|22.19 (20.23)|21.05 (19.40)| **20.39 (18.29)** |
| HCGNet-A1 | 0.15G|22.46|18.98 (17.86) | 19.04 (18.35)|22.30 (21.64)|21.54 (20.97)|**18.72 (17.54)** |

- `Ens` : Ensemble performance with retaining all peer networks.


## Perform experiments on ImageNet dataset

### Dataset preparation

- Download the ImageNet dataset to YOUR_IMAGENET_PATH and move validation images to labeled subfolders
    - The [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) may be helpful.

- Create a datasets subfolder and a symlink to the ImageNet dataset

```
$ ln -s PATH_TO_YOUR_IMAGENET ./data/
```
Folder of ImageNet Dataset:
```
data/ImageNet
├── train
├── val
```

### Training for baseline
```
python main_imagenet_baseline.py --arch resnet34 --gpu 0
```

### Training by MCL-OKD
```
python main_imagenet_mcl_okd.py --arch mcl_okd_resnet34 --gpu 0
```

| Baseline | MCL-OKD | MCL-OKD (Ens) | 
| - | - |- |
| 25.43| 24.64 |**23.26**|

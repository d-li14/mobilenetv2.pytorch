# PyTorch Implemention of MobileNet V2

<h2>

```diff
+ Release of next generation of MobileNet in my repo *mobilenetv3.pytorch*
+ Release of advanced design of MobileNetV2 in my repo *HBONet* [ICCV 2019]
+ Release of better pre-trained model. See below for details.
```

</h2>

Reproduction of MobileNet V2 architecture as described in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov and Liang-Chieh Chen on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework.

This implementation provides an example procedure of training and validating any prevalent deep neural network architecture, with modular data processing, training, logging and visualization integrated.

# Requirements
## Dependencies
* PyTorch 1.0+
* [NVIDIA-DALI](https://github.com/NVIDIA/DALI) (in development, not recommended)
## Dataset
Download the ImageNet dataset and move validation images to labeled subfolders.
To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# Pretrained models
The pretrained [MobileNetV2 1.0](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2-c5e733a8.pth) achieves **72.834% top-1 accuracy** and 91.060% top-5 accuracy on ImageNet validation set, which is higher than the statistics reported in the original paper and official [TensorFlow implementation](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

### MobileNetV2 with a spectrum of width multipliers
| Architecture      | # Parameters | MFLOPs | Top-1 / Top-5 Accuracy (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| [MobileNetV2 1.0](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_1.0-0c6065bc.pth)    | 3.504M | 300.79 | 72.192 / 90.534 |
| [MobileNetV2 0.75](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.75-dace9791.pth)  | 2.636M | 209.08 | 69.952 / 88.986 |
| [MobileNetV2 0.5](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth)    | 1.968M | 97.14 | 64.592 / 85.392 |
| [MobileNetV2 0.35](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.35-b2e15951.pth)  | 1.677M |     59.29 | 60.092 / 82.172  |
| [MobileNetV2 0.25](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.25-b61d2159.pth)  | 1.519M |     37.21 | 52.352 / 75.932  |
| [MobileNetV2 0.1](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.1-7d1d638a.pth)    | 1.356M | 12.92 | 34.896 / 56.564 |

### MobileNetV2 1.0 with a spectrum of input resolutions
| Architecture      | # Parameters | MFLOPs | Top-1 / Top-5 Accuracy (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| [MobileNetV2 224x224](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_1.0-0c6065bc.pth)    | 3.504M | 300.79 | 72.192 / 90.534 |
| [MobileNetV2 192x192](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_192x192-e423d99e.pth)| 3.504M | 221.33 | 71.076 / 89.760 |
| [MobileNetV2 160x160](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_160x160-64dc7fa1.pth)| 3.504M |  154.10 | 69.504 / 88.848 |
| [MobileNetV2 128x128](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_128x128-fd66a69d.pth)| 3.504M |  99.09 | 66.740 / 86.952 |
| [MobileNetV2 96x96](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_96x96-ff0e83d8.pth)    | 3.504M |  56.31 | 62.696 / 84.046 |

Taking MobileNetV2 1.0 as an example, pretrained models can be easily imported using the following lines and then finetuned for other vision tasks or utilized in resource-aware platforms.

```python
from models.imagenet import mobilenetv2

net = mobilenetv2()
net.load_state_dict(torch.load('pretrained/mobilenetv2-c5e733a8.pth'))
```

# Usage
## Training
Configuration to reproduce our strong results efficiently, consuming around 2 days on 4x TiTan XP GPUs with [non-distributed DataParallel](https://pytorch.org/docs/master/nn.html#torch.nn.DataParallel) and [PyTorch dataloader](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).
* *batch size* 256
* *epoch* 150
* *learning rate* 0.05
* *LR decay strategy* cosine
* *weight decay* 0.00004

The [newly released model](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2-c5e733a8.pth) achieves even higher accuracy, with larger bacth size (1024) on 8 GPUs, higher initial learning rate (0.4) and longer training epochs (250). In addition, a dropout layer with the dropout rate of 0.2 is inserted before the final FC layer, no weight decay is imposed on biases and BN layers and the learning rate ramps up from 0.1 to 0.4 in the first five training epochs.

```shell
python imagenet.py \
    -a mobilenetv2 \
    -d <path-to-ILSVRC2012-data> \
    --epochs 150 \
    --lr-decay cos \
    --lr 0.05 \
    --wd 4e-5 \
    -c <path-to-save-checkpoints> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -j <num-workers>
```

## Test
```shell
python imagenet.py \
    -a mobilenetv2 \
    -d <path-to-ILSVRC2012-data> \
    --weight <pretrained-pth-file> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -e
```

# Citations
The following is a [BibTeX](www.bibtex.org) entry for the MobileNet V2 paper that you should cite if you use this model.
```
@InProceedings{Sandler_2018_CVPR,
author = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
title = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
If you find this implementation helpful in your research, please also consider citing:
```
@InProceedings{Li_2019_ICCV,
author = {Li, Duo and Zhou, Aojun and Yao, Anbang},
title = {HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```

# License
This repository is licensed under the [Apache License 2.0](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/LICENSE).

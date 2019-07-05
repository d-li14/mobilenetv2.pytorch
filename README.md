# PyTorch Implemention of MobileNet V2
<h2>

```diff
+ Release of next generation of MobileNet in my repo *mobilenetv3.pytorch*
```

</h2>

Reproduction of MobileNet V2 architecture as described in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov and Liang-Chieh Chen on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework.

This implementation provides an example procedure of training and validating any prevalent deep neural network architecture, with modular data processing, training, logging and visualization integrated.

# Requirements
## Dataset
Download the ImageNet dataset and move validation images to labeled subfolders.
To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# Pretrained models
The pretrained MobileNetV2 1.0 achieves **72.192% top-1 accuracy** and 90.534% top-5 accuracy on ImageNet validation set, which is higher than the statistics reported in the original paper and official [TensorFlow implementation](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

### MobileNetV2 with a spectrum of width multipliers
| Architecture      | # Parameters | MFLOPs | Top-1 / Top-5 Accuracy (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| [MobileNetV2 1.0](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_1.0-0c6065bc.pth)    | 3.504M | 300.79 | 72.192 / 90.534 |
| [MobileNetV2 0.75](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.75-dace9791.pth)  | 2.636M | 209.08 | 69.952 / 88.986 |
| [MobileNetV2 0.5](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth)    | 1.968M | 97.14 | 64.592 / 85.392 |
| [MobileNetV2 0.35](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.35-b2e15951.pth)  | 1.677M |     59.29 | 60.092 / 82.172  |
| [MobileNetV2 0.25](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.25-b61d2159.pth)  | 1.519M |     37.21 | 52.352 / 75.932  |
| [MobileNetV2 0.1](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.1-7d1d638a.pth)    | 1.356M | 12.92 | 34.896 / 56.564 |

*Note: Channels of MobileNetV1 0.1 are set to be divisible by 4 while the default number is 8*

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
net.load_state_dict(torch.load('pretrained/mobilenetv2_1.0-0c6065bc.pth'))
```

The command below shows an example which loads and evaluates the half-width MobileNetV2 pretrained model:
```
python imagenet.py \
	-d ~/path/to/ILSVRC2012/ -j16 -e \
	--weightfile pretrained/mobilenetv2_0.5-eaa6f9ad.pth \
	--arch="mobilenetv2" \
	--gpu-id="0,1,2,3" \
	--width_mult 0.5 \
	--input_size 224
```

# Training

The command below trains a 1.0 width MobileNetV2 model with an input resolution of 224 from scratch:
```
python imagenet.py \
	-d ~/path/to/ILSVRC2012/ -j16 \
	--arch="mobilenetv2" \
	--gpu-id="0,1,2,3" \
	--epochs=300 \
	--lr=0.045 \
	--lr-decay="linear2exp" \
	--gamma=0.98 \
	--weight-decay=0.00004
	--width_mult 1.0 \
	--input_size 224 \
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

# License
This repository is licensed under the [Apache License 2.0](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/LICENSE).

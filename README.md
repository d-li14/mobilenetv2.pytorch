## Release of the next generation of MobileNets in [mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)
# PyTorch Implemention of MobileNet V2
Reproduction of MobileNet V2 architecture as described in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov and Liang-Chieh Chen on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework. Adapted from [pytorch-classification](https://github.com/bearpaw/pytorch-classification) and [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2).

This implementation provides an example procedure of training and validating any prevalent deep neural network architecture, with modular data processing, training, logging and visualization integrated.

# Requirements
## Dataset
Download the ImageNet dataset and move validation images to labeled subfolders.
To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# Pretrained models
[Our pretrained model](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2-0c6065bc.pth) achieves **72.192% top-1 accuracy** and 90.534% top-5 accuracy on ImageNet validation set, which is higher than the statistics reported in the original paper and official [TensorFlow implementation](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

Pretrained model can be easily imported using the following lines and then finetuned for other vision tasks or utilized in resource-limited platforms.
```python
from models.imagenet import mobilenetv2

net = mobilenetv2()
net.load_state_dict(torch.load('pretrained/mobilenetv2-0c6065bc.pth'))
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

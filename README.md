# Decoupled-Contrastive-Learning
This repository is an implementation for the loss function proposed in [Decoupled Contrastive Loss](https://arxiv.org/pdf/2110.06848.pdf) paper.

## Requirements

- Pytorch
- Numpy

## Usage Example

```python
import torch
import torchvision.models as models

from loss import dcl

resnet18 = models.resnet18()
random_input = torch.rand((10, 3, 244, 244))
output = resnet18(random_input)

# for DCL
loss_fn = dcl.DCL(temperature=0.5)
loss = loss_fn(output, output)  # loss = tensor(-0.2726, grad_fn=<AddBackward0>

# for DCLW
loss_fn = dcl.DCLW(temperature=0.5, sigma=0.5)
loss = loss_fn(output, output)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)
```

## Results

| Loss          | 32 Batch Size | 64 Batch Size |
| ------------- | ------------- | ------------- |
| Cross Entropy | 78.3          | 81.47         |
| DCL           | 84.6          | 85.57         |
| DCLW          | 83.32         | 82.68         |


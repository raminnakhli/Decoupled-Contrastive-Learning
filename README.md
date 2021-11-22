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

## How to Run

You can simply `run_simclr.sh` to train and test the SimCLR. 

You can select different loss function by changing the `LOSS` variable in the same file. The valid value for this variable include `ce` for Cross-Entropy, `dcl` for Decoupled Contrastive Loss and `dclw` for Weighted Decoupled Contrastive Loss.

The dataset can be selected by changing the `DATASET` variable as well. Valid values include `cifar10`, `cifar100`, and `stl10`.

The final results can be seen in the output stream.

Also, if you operate under Slurm, you can submit a job using `sbatch run_simclr.sh` and find the output file in the same directory.



## Results

Below are the SimCLR results of the Resnet18 on the CIFAR10 dataset. The temperature is set to 0.1 and sigma to 0.5. The model is also trained for 100 epochs on SimCLR pretraining and 100 epochs on linear probing.

| Loss          | 32 Batch Size | 64 Batch Size | 128 Batch Size | 256 Batch Size |
| ------------- | ------------- | ------------- | -------------- | -------------- |
| Cross Entropy | 78.3          | 81.47         | 83.09          | 83.26          |
| DCL           | 84.6          | 85.57         | 85.63          | 85.36          |
| DCLW          | 83.32         | 82.68         | 83.5           | 82.7           |

Below are the SimCLR results of the Resnet18 on the CIFAR100 dataset. The temperature is set to 0.1 and sigma to 0.5. The model is also trained for 100 epochs on SimCLR pretraining and 100 epochs on linear probing.

| Loss          | 32 Batch Size | 64 Batch Size | 128 Batch Size | 256 Batch Size |
| ------------- | ------------- | ------------- | -------------- | -------------- |
| Cross Entropy | 52.2          | 57.76         | 58             | 59.6           |
| DCL           | 60.1          | 60.33         | 61.73          | 61.92          |
| DCLW          | 60.16         | 59.7          | 59.05          | 57.3           |

## Credits

- The SimCLR implementation is taken from https://github.com/leftthomas/SimCLR.

## Contribution

Any contributions would be appreciated!

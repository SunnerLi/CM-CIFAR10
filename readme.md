# CM-CIFAR10

Train consistency models from scratch and sample with 1 step on CIFAR-10. Code is written in PyTorch. 

Usage
---
```cmd
### Train
$ python train.py --output_folder checkpoint/cifar10-cm
### Sampling
$ python sample.py --model_path checkpoint/cifar10-cm/last.pth
```

Ref
---
[Official repository](https://github.com/openai/consistency_models/tree/main)
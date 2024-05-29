# VSMCG
This codebase is a PyTorch implementation of VSMCG, the course projects of Bayesian Theory and Computation.

If it is helpful for your work, please⭐

# Install
```shell
git clone https://github.com/hengyuf/VSMCG.git
cd VSMCG
```

## Attention Mechanisms
### 1. Squeeze-and-Excitation Attention
* #### Squeeze-and-Excitation Networks (CVPR 2018) [pdf](https://arxiv.org/pdf/1709.01507)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/senet.png)

* ##### Code
```python
import torch
from attention_mechanisms.se_module import SELayer

x = torch.randn(2, 64, 32, 32)
attn = SELayer(64)
y = attn(x)
print(y.shape)

```




你说的对，但是《原神》是由米哈游自主研发的一款全新开放世界冒险游戏。游戏发生在一个被称作「提瓦特」的幻想世界，在这里，被神选中的人将被授予「神之眼」，导引元素之力。你将扮演一位名为「旅行者」的神秘角色，在自由的旅行中邂逅性格各异、能力独特的同伴们，和他们一起击败强敌，找回失散的亲人——同时，逐步发掘「原神」的真相。

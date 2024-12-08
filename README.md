# Ctrl_C_NN
Dependency-free neural network inference framework in a single file

## What is it for
Inference with simple neural networks where installing dependencies is not possible. This project is and will be dependency-free and has the most open open-source license. Whatever you need it for, just copy the single .py file into your project, and you can run an already-trained neural network. 

## What is it NOT for
Since it is written 100% in Python, its performance is terrible compared to PyTorch or numpy-based frameworks. It's not designed for the training of neural networks but to load and run simple Pytorch neural networks.

## WIP
| Description                              | Status                 |
|------------------------------------------|------------------------|
| Base Tensor class                        | :white_check_mark:     |
| Tensor operations (+, *, @)              | :white_check_mark:     |
| Tensor Broadcasting                      | :white_check_mark:     |
| Tensor Shape Manipulation (e.g. reshape) | :white_check_mark:     |
| Simple Layers and Non-linearities        | :large_orange_diamond: |
| Forward pass of simple NN                | :x:                    |
| Backward pass of simple NN               | :x:                    |
| Convolutional Layers                     | :x:                      |
| Reading pth files                        | :x:                    |
| Basic Image I/O                          | :x:                    |
| ...                                      | :x:                    |
| ...                                      | :x:                    |


## Sample Usage Tensor
```python
from ctrl_c_nn import Tensor

a = Tensor.zeros((2, 4, 8, 2))
b = Tensor.zeros((2, 8))
c = a@b  # shape (2, 4, 8, 8)
d = c[0, 2:, :, :1] + b.unsqueeze(2)  # shape (2,8,1)
e = d.reshape((1,2,4,2,1)) + 1  # shape (1,2,4,2,1)
f = e.sum(3)  # shape (1,2,4,1)
g = e.permute((3,0,2,1)) # shape (1, 1, 4, 2)
```
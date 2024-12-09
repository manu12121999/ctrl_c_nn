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
| Simple Layers and Non-linearities        | :white_check_mark:     |
| Forward pass of simple NN                | :large_orange_diamond: |
| Backward pass of simple NN               | :large_orange_diamond: |
| Convolutional Layers                     | :x:                    |
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

## Sample Usage NN
from ctrl_c_nn import nn, Tensor

model = nn.Sequential(
    nn.Linear(20, 128),
    nn.LeakyReLU(),
    nn.SkipStart("a"),
    nn.Linear(128, 128),
    nn.LeakyReLU(),
    nn.SkipEnd("a"),
    nn.Linear(128, 2),
    nn.LeakyReLU(),
)
loss_fn = nn.MSELoss()

for i in range(2000):
    input_tensor = Tensor.random_float((8, 20))
    target_tensor = Tensor.ones(output_tensor.shape)

    #  no zero_grad() atm (grads dont accumulate)
    output_tensor = model(input_tensor)
    loss = loss_fn(output_tensor, target_tensor)

    print("loss", loss.item(), "          iteration", i)

    dout = loss_fn.backward(loss)
    dout = model.backward(dout)
    model.update(lr=0.001)


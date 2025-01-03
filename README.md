# Ctrl_C_NN: A Dependency-Free Python Neural Network Framework.

## Key Features
- Single-file framework, dependency-free
- Simple inference for pretrained PyTorch models

## What is it for
Inference with simple neural networks where installing dependencies is not possible. This project is and will be dependency-free and has the most open open-source license. Whatever you need it for, just copy the single .py file into your project, and you can run an already-trained neural network. 

Also useful for educational or academic purposes since the core functionality can be easily understood (or changed).
## What is it NOT for
Since it is written 100% in Python, its performance is terrible compared to PyTorch (~500x slower). It's not designed for the training of neural networks but to load and run simple pretrained Pytorch NNs.

## Sample Usage: Inference with pretrained PyTorch NN
```python
import ctrl_c_nn
from ctrl_c_nn import Tensor, nn, F, ImageIO

input_image = ImageIO.read_png("dog.png", num_channels=3, resize=(224, 224), 
                               dimorder="BCHW", to_float=True, 
                               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = SqueezeNet()  # when defining change all torch.nn to ctrl_c_nn.nn
model.load_state_dict(ctrl_c_nn.load("model.pth"))
output = model(input_image)
probabilities = F.softmax(output[0], dim=0)
```

## WIP
CURRENTLY IN DEVELOPMENT

| Description                              | Status                     |
|------------------------------------------|----------------------------|
| Base Tensor class                        | :white_check_mark:         |
| Tensor operations (+, *, @)              | :white_check_mark:         |
| Tensor Broadcasting                      | :white_check_mark:         |
| Tensor Shape Manipulation (e.g. reshape) | :white_check_mark:         |
| Simple Layers and Non-linearities        | :white_check_mark:         |
| Forward pass of simple NN                | :white_check_mark:         |
| Backward pass of simple NN               | :large_orange_diamond: WIP |
| Convolutional Layers                     | :white_check_mark:         |
| Transposed Conv & Upsampling             | :large_orange_diamond: WIP |
| Transformer Layers                       | :x:                        |
| Reading pth files                        | :white_check_mark:         |
| Forward pass of CNN                      | :white_check_mark:         |
| Backward pass of CNN                     | :x:                        |
| Image IO: Read PNG files                 | :white_check_mark:         |
| Image IO: Read JPG files                 | :x:                        |
| Image IO: Save images                    | :x:                        |
| ...                                      | :x:                        |
| ...                                      | :x:                        |

Hopefully one day

| Description              | Status |
|--------------------------|--------|
| GPU Matmul (e.g. OpenCL) | :x:    |
| Autograd                 | :x:    |
| ...                      | :x:    |




## Sample Usage: Tensor
```python
from ctrl_c_nn import Tensor

a = Tensor.zeros(2, 4, 8, 2)
b = Tensor.zeros((2, 8))
c = a@b  # shape (2, 4, 8, 8)
d = c[0, 2:, :, :1] + b.unsqueeze(2)  # shape (2,8,1)
e = d.reshape((1,2,4,2,1)) + 1  # shape (1,2,4,2,1)
f = e.sum(3)  # shape (1,2,4,1)
g = e.permute((3,0,2,1)) # shape (1, 1, 4, 2)
```

## Sample Usage: Training a simple NN
```python
from ctrl_c_nn import nn, Tensor

# it's the simplest to define the network as one Sequential
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
    target_tensor = Tensor.fill(output_tensor.shape, 1.0)

    #  no zero_grad() atm (grads dont accumulate)
    output_tensor = model(input_tensor)
    loss = loss_fn(output_tensor, target_tensor)

    print("loss", loss.item(), " iteration", i)

    dout = loss_fn.backward(loss)
    dout = model.backward(dout)
    model.update(lr=0.001)
```
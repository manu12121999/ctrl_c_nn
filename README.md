# Ctrl_C_NN
Dependency-free neural network inference framework in a single file

## What is it for
Inference with simple neural networks where installing dependencies is not possible. This project is and will be dependency-free and has the most open open-source license. Whatever you need it for, just copy the single .py file into your project, and you can run an already-trained neural network. 

## What is it NOT for
Since it is written 100% in Python, its performance is terrible compared to PyTorch or numpy-based frameworks. It also does not (and never will) support the training of neural networks but is designed to load simple Pytorch neural networks.

## WIP
| Description                              | Status          |
|------------------------------------------|-----------------|
| Base Tensor class                        | :white_check_mark: |
| Tensor operations (+, *, @)              | :x:             |
| Tensor Broadcasting                      | :x:             |
| Simple Layers and Non-linearities        | :x:             |
| Forward pass of simple NN                | :x:             |
| Reading pth files                        | :x:             |
| Basic Image I/O                          | :x:             |
| ...                                      | :x:             |
| ...                                      | :x:             |

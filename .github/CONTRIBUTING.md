# What should be considered when contributing:
1. Don't add dependencies
2. Keep the code short and readable. Don't overcomment everything
3. Do not copy code or use AI to code since this project has a really loose copyright


# Useful contributions include

1. Add new Layers
2. Implement more parameter options for existing layers (e.g. dilated convolutions)
3. Speed up inference time
4. Documentation / Test


## New Layers
The readme give a small overview about which layers are still missing. 
Not every feature has to be implemented, but the using the layers should 
either give the same result as the pytorch equivalent or raise a NotImplementedError

## Speedup
Any ideas to make matrix-matrix multiplications faster are highly welcome.
Mainly the case `A@B.T` where A and B are two-dimensional is important, as this is used for Convolutional and Linear Layers.
To report speedup, please time the inference time of `SqueezeNet_ctrlc.py` using python 3.12 or newer
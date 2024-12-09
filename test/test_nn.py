import unittest
from ctrl_c_nn import Tensor, nn


class TestNNForward(unittest.TestCase):
    def test_shape_forward_linear(self):
        input = Tensor.random_float((8, 128))
        linear_layer = nn.Linear(128, 256)
        pred = linear_layer(input)
        self.assertEqual(pred.shape, (8, 256), f"Shape is wrong after linear layer")

    def test_shape_forward_relu(self):
        input = Tensor.random_float((8, 128))
        relu = nn.ReLU()
        pred = relu(input)
        self.assertEqual(pred.shape, (8, 128), f"Shape is wrong after relu layer.")

    def test_shape_forward_lin_lin(self):
        input = Tensor.random_float((8, 64))
        linear_layer1 = nn.Linear(64, 96)
        linear_layer2 = nn.Linear(96, 128)
        pred = linear_layer2(linear_layer1(input))
        self.assertEqual(pred.shape, (8, 128), f"Shape is wrong after two linear layer.")

    def test_shape_forward_seq(self):
        input = Tensor.random_float((8, 64))
        model = nn.Sequential(
                nn.Linear(64, 96),
                nn.Linear(96, 128)
                )
        pred = model(input)
        self.assertEqual(pred.shape, (8, 128), f"Shape is wrong after sequential layer.")

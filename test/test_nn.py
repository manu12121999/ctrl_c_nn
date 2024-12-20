import unittest
import numpy as np
import ctrl_c_nn

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TestNNForward(unittest.TestCase):
    def test_shape_forward_linear(self):
        input = ctrl_c_nn.Tensor.random_float((8, 128))
        linear_layer = ctrl_c_nn.nn.Linear(128, 256)
        pred = linear_layer(input)
        self.assertEqual(pred.shape, (8, 256), f"Shape is wrong after linear layer")

    def test_shape_forward_relu(self):
        input = ctrl_c_nn.Tensor.random_float((8, 128))
        relu = ctrl_c_nn.nn.ReLU()
        pred = relu(input)
        self.assertEqual(pred.shape, (8, 128), f"Shape is wrong after relu layer.")

    def test_shape_forward_lin_lin(self):
        input = ctrl_c_nn.Tensor.random_float((8, 64))
        linear_layer1 = ctrl_c_nn.nn.Linear(64, 96)
        linear_layer2 = ctrl_c_nn.nn.Linear(96, 128)
        pred = linear_layer2(linear_layer1(input))
        self.assertEqual(pred.shape, (8, 128), f"Shape is wrong after two linear layer.")

    def test_shape_forward_seq(self):
        input = ctrl_c_nn.Tensor.random_float((8, 64))
        model = ctrl_c_nn.nn.Sequential(
                ctrl_c_nn.nn.Linear(64, 96),
                ctrl_c_nn.nn.Linear(96, 128)
                )
        pred = model(input)
        self.assertEqual(pred.shape, (8, 128), f"Shape is wrong after sequential layer.")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch is not installed")
class TestNNLayers(unittest.TestCase):

    def test_conv_layer_single_batch(self):
        input_tensor = np.random.randn(1, 6, 16, 16)
        weights = np.random.randn(9, 6, 3, 3)
        bias = np.random.randn(9)

        # Pytorch
        m_pytorch = torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1)
        m_pytorch.weight = torch.nn.Parameter(torch.tensor(weights))
        m_pytorch.bias = torch.nn.Parameter(torch.tensor(bias))
        out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1)
        m_ctrl_c.weight.replace(ctrl_c_nn.Tensor(weights))
        m_ctrl_c.bias.replace(ctrl_c_nn.Tensor(bias))
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)


    def test_conv_layer(self):
        input_tensor = np.random.randn(3, 6, 48, 16)
        weights = np.random.randn(54, 6, 3, 3)
        bias = np.random.randn(54)

        # Pytorch
        m_pytorch = torch.nn.Conv2d(in_channels=6, out_channels=54, kernel_size=3, stride=1, padding=1)
        m_pytorch.weight = torch.nn.Parameter(torch.tensor(weights))
        m_pytorch.bias = torch.nn.Parameter(torch.tensor(bias))
        out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.Conv2d(in_channels=6, out_channels=54, kernel_size=3, stride=1, padding=1)
        m_ctrl_c.weight.replace(ctrl_c_nn.Tensor(weights))
        m_ctrl_c.bias.replace(ctrl_c_nn.Tensor(bias))
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)

    def test_grouped_conv_layer(self):
        input_tensor = np.random.randn(3, 6, 48, 16)
        weights = np.random.randn(54, 3, 3, 3)
        bias = np.random.randn(54)

        # Pytorch
        m_pytorch = torch.nn.Conv2d(in_channels=6, out_channels=54, kernel_size=3, stride=1, padding=1, groups=2)
        m_pytorch.weight = torch.nn.Parameter(torch.tensor(weights))
        m_pytorch.bias = torch.nn.Parameter(torch.tensor(bias))
        out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.Conv2d(in_channels=6, out_channels=54, kernel_size=3, stride=1, padding=1, groups=2)
        m_ctrl_c.weight.replace(ctrl_c_nn.Tensor(weights))
        m_ctrl_c.bias.replace(ctrl_c_nn.Tensor(bias))
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)

    def test_linear_layer(self):
        input_tensor = np.random.randn(3, 543)
        weights = np.random.randn(54, 543)
        bias = np.random.randn(54)

        # Pytorch
        m_pytorch = torch.nn.Linear(in_features=6, out_features=54, bias=True)
        m_pytorch.weight = torch.nn.Parameter(torch.tensor(weights))
        m_pytorch.bias = torch.nn.Parameter(torch.tensor(bias))
        out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.Linear(in_features=6, out_features=54, bias=True)
        m_ctrl_c.weight.replace(ctrl_c_nn.Tensor(weights))
        m_ctrl_c.bias.replace(ctrl_c_nn.Tensor(bias))
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)


    def test_MaxPool_layer(self):
        for kernel_size in range(1, 4):
            for stride in range(1, 4):
                for padding in range(0,kernel_size//2):
                    input_tensor = np.random.randn(3, 6, 42, 42)

                    # Pytorch
                    m_pytorch = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                    out_pytorch = m_pytorch(torch.tensor(input_tensor))

                    # Ctrl_C
                    m_ctrl_c = ctrl_c_nn.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                    out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

                    self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
                    mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
                    self.assertLess(mean_diff.item(), 1e-5)
                    max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
                    self.assertLess(max_diff.item(), 1e-2)

    def test_ReLU_layer(self):
        input_tensor = np.random.randn(3, 6, 42, 42)
        # Pytorch
        m_pytorch = torch.nn.ReLU()
        out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.ReLU()
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)

    def test_LeakyReLU_layer(self):
        input_tensor = np.random.randn(3, 6, 42, 42)
        # Pytorch
        m_pytorch = torch.nn.LeakyReLU(negative_slope=0.2)
        out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.LeakyReLU(negative_slope=0.2)
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)

    def test_Dropout_layer(self):
        input_tensor = np.random.randn(3, 6, 42, 42)
        # Pytorch
        m_pytorch = torch.nn.Dropout(p=0.2)
        m_pytorch.eval()
        out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.Dropout(p=0.2)
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)

    def test_BN_layer(self):
        input_tensor = np.random.randn(1, 17, 48, 16)
        weights = np.random.randn(17)
        bias = np.random.randn(17)
        running_mean = np.random.randn(17)
        running_var = np.abs(np.random.randn(17))

        # Pytorch
        with torch.no_grad():
            m_pytorch = torch.nn.BatchNorm2d(num_features=17, eps=1e-04).eval()
            m_pytorch.weight = torch.nn.Parameter(torch.tensor(weights))
            m_pytorch.bias = torch.nn.Parameter(torch.tensor(bias))
            m_pytorch.running_mean = torch.nn.Parameter(torch.tensor(running_mean))
            m_pytorch.running_var = torch.nn.Parameter(torch.tensor(running_var))
            out_pytorch = m_pytorch(torch.tensor(input_tensor))

        # Ctrl_C
        m_ctrl_c = ctrl_c_nn.nn.BatchNorm2d(num_features=17, eps=1e-04)
        m_ctrl_c.weight.replace(ctrl_c_nn.Tensor(weights))
        m_ctrl_c.bias.replace(ctrl_c_nn.Tensor(bias))
        m_ctrl_c.running_mean.replace(ctrl_c_nn.Tensor(running_mean))
        m_ctrl_c.running_var.replace(ctrl_c_nn.Tensor(running_var))
        out_ctrl_c = m_ctrl_c(ctrl_c_nn.Tensor(input_tensor))

        self.assertEqual(out_pytorch.shape, out_ctrl_c.shape, "shape mismatch")
        mean_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().mean()
        self.assertLess(mean_diff.item(), 1e-5)
        max_diff = (out_pytorch - torch.tensor(out_ctrl_c.tolist())).abs().max()
        self.assertLess(max_diff.item(), 1e-2)
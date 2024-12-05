import unittest
import numpy as np
import torch

shapes = [
    (2,),
    (3,),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (3, 3, 3),
    (3, 2, 3),
    (3, 2, 4),
    (3, 4, 2),
    (2, 4, 2),
    (2, 2, 2),
    (2, 2, 4),
    (2, 2, 4),
    (1, 2, 2, 4),
    (1, 2, 2, 4),
    (2, 3, 2, 4),
    (3, 3, 4, 2),
    (4, 3, 2, 1),
    (2, 3, 2, 1),
    (4, 2, 3, 2, 1),
    (4, 2, 3, 1, 2),
    (3, 2, 3, 1, 2),
]


class TestMatmul(unittest.TestCase):

    def generic_various_shapes_test(self, func1, func2):
        # Testing various tensor shapes
        for i in range(len(shapes)):
            for j in range(len(shapes)):
                a_np = np.random.randint(shapes[i])
                b_np = np.random.randint(shapes[j])
                a_torch, b_torch = torch.tensor(a_np), torch.tensor(b_np)

                try:
                    result_np = func1(a_np, b_np)
                except ValueError:
                    result_np = np.array([])
                try:
                    result_torch = func2(a_torch, b_torch)
                except RuntimeError:
                    result_torch = torch.Tensor([])

                result_np_list = result_np.tolist()
                result_torch_list = result_torch.tolist()

                self.assertEqual(result_np_list, result_torch_list, f"Tensor operation {func2} failed for shapes {a_np.shape} and {b_np.shape}.")

    def test_matmul_various_shapes(self):
        self.generic_various_shapes_test(np.matmul, torch.matmul)

    def test_add_various_shapes(self):
        self.generic_various_shapes_test(np.add, torch.add)
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
import ctrl_c_nn as ccm


small_shapes = [
    (2,),
    (3,),
    (2, 1),
    (1, 2),
    (2, 3),
    (3, 2),
    (3, 3),
]

small_shapes_non_singleton = [
    (2,),
    (3,),
    (2, 3),
    (3, 2),
    (3, 3),
]

big_shapes_non_singleton = [
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
    (2, 3, 2, 4),
    (3, 3, 4, 2),
    (4, 2, 3, 2, 4),
    (4, 2, 3, 2, 2),
    (3, 2, 3, 2, 2),
]

big_shapes = [
    (2, 3),
    (3, 2),
    (3, 3),
    (1, 3),
    (3, 1),
    (3, 3, 3),
    (3, 2, 3),
    (3, 2, 4),
    (3, 4, 2),
    (2, 4, 2),
    (2, 2, 2),
    (2, 2, 4),
    (2, 2, 4),
    (3, 3, 3, 1),
    (1, 2, 2, 4),
    (1, 2, 2, 4),
    (2, 3, 2, 4),
    (2, 3, 1, 4),
    (3, 3, 4, 2),
    (4, 3, 2, 1),
    (2, 3, 2, 1),
    (4, 2, 3, 2, 1),
    (4, 2, 3, 1, 2),
    (3, 2, 3, 1, 2),
    (2, 1, 3, 2, 4),
]

all_shapes = [
    (1,),
    (2,),
    (3,),
    (2, 1),
    (1, 2),
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
    (3, 3, 3, 1),
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


class TestTensorLinOps(unittest.TestCase):

    def generic_various_shapes_test(self, func1, func2, init2, shapes):
        # Testing various tensor shapes
        for i in range(len(shapes)):
            for j in range(len(shapes)):
                mat1 = np.random.randint(0, 10, shapes[i])
                mat2 = np.random.randint(0, 10, shapes[j])

                mat1_B, mat2_B = init2(mat1), init2(mat2)
                print("MAT A", mat1_B)
                print("MAT B", mat2_B)
                try:
                    result_A = func1(mat1, mat2).tolist()
                    print(f"RES1", result_A)
                except (ValueError, AssertionError, RuntimeError):
                    print("No RES1")
                    result_A = None
                try:
                    result_B = func2(mat1_B, mat2_B).tolist()
                    print("RES2", result_B)
                except (ValueError, AssertionError, RuntimeError):
                    print("No RES2")
                    result_B = None

                self.assertEqual(result_A, result_B, f"Tensor operation {func2} failed for shapes {mat1.shape} and {mat2.shape}. Elemes {mat1} and {mat2}")

    @unittest.skip
    def test_matmul_various_shaped_sanity(self):
        self.generic_various_shapes_test(func1=np.matmul, func2=torch.matmul, init2=lambda x: torch.Tensor(x).to(torch.int32))

    @unittest.skip
    def test_add_various_shaped_sanity(self):
        self.generic_various_shapes_test(np.add, torch.add, init2=lambda x: torch.Tensor(x).to(torch.int32))

    def test_basic_add(self):
        mat1 = np.random.randint(0, 10, size=(400, 200))
        mat2 = np.random.randint(0, 10, size=(400, 200))
        result_A = mat1 + mat2
        result_B = ccm.Tensor(mat1.tolist()) + ccm.Tensor(mat2.tolist())
        self.assertEqual(result_A.tolist(), result_B.tolist(),f"Tensor operation basic add failed.")

    def test_basic_mul(self):
        mat1 = np.random.randint(0, 10, size=(400, 200))
        mat2 = np.random.randint(0, 10, size=(400, 200))
        result_A = mat1 * mat2
        result_B = ccm.Tensor(mat1.tolist()) * ccm.Tensor(mat2.tolist())
        self.assertEqual(result_A.tolist(), result_B.tolist(),f"Tensor operation basic mul failed.")

    def test_basic_matmul(self):
        mat1 = np.random.randint(0, 10, size=(400, 200))
        mat2 = np.random.randint(0, 10, size=(200, 300))
        result_A = mat1 @ mat2
        result_B = ccm.Tensor(mat1.tolist()) @ ccm.Tensor(mat2.tolist())
        self.assertEqual(result_A.tolist(), result_B.tolist(),f"Tensor operation basic matmul failed.")

    def test_matmul_broadcast(self):
        mat1 = np.random.randint(0, 10, size=(30, 40, 200))
        mat2 = np.random.randint(0, 10, size=(200, 30))
        result_A = mat1 @ mat2
        result_B = ccm.Tensor(mat1.tolist()) @ ccm.Tensor(mat2.tolist())
        self.assertEqual(result_A.tolist(), result_B.tolist(),f"Tensor operation broadcast matmul failed.")

    def test_matmul_batched(self):
        mat1 = np.random.randint(0, 10, size=(4, 40, 200))
        mat2 = np.random.randint(0, 10, size=(4, 200, 30))
        result_A = mat1 @ mat2
        result_B = ccm.Tensor(mat1.tolist()) @ ccm.Tensor(mat2.tolist())
        self.assertEqual(result_A.tolist(), result_B.tolist(),f"Tensor operation broadcast matmul failed.")

    def test_matmul_various_shaped_int(self):
        self.generic_various_shapes_test(np.matmul, ccm.Tensor.__matmul__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=all_shapes)

    def test_add_various_shaped_int(self):
        self.generic_various_shapes_test(np.add, ccm.Tensor.__add__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=all_shapes)

    def test_matmul_small_shaped_int(self):
        self.generic_various_shapes_test(np.matmul, ccm.Tensor.__matmul__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=small_shapes)

    def test_add_small_shaped_int(self):
        self.generic_various_shapes_test(np.add, ccm.Tensor.__add__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=small_shapes)

    def test_matmul_small_non_sing_shaped_int(self):
        self.generic_various_shapes_test(np.matmul, ccm.Tensor.__matmul__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=small_shapes_non_singleton)

    def test_add_small_non_sing_shaped_int(self):
        self.generic_various_shapes_test(np.add, ccm.Tensor.__add__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=small_shapes_non_singleton)

    def test_matmul_big_shaped_int(self):
        self.generic_various_shapes_test(np.matmul, ccm.Tensor.__matmul__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=big_shapes)

    def test_add_big_shaped_int(self):
        self.generic_various_shapes_test(np.add, ccm.Tensor.__add__, init2=lambda x: ccm.Tensor(x.tolist()), shapes=big_shapes)

    def test_matmul_big_non_sing_shaped_int(self):
        self.generic_various_shapes_test(np.matmul, ccm.Tensor.__matmul__, init2=lambda x: ccm.Tensor(x.tolist()),
                                         shapes=big_shapes_non_singleton)

    def test_add_big_non_sing_shaped_int(self):
        self.generic_various_shapes_test(np.add, ccm.Tensor.__add__, init2=lambda x: ccm.Tensor(x.tolist()),
                                         shapes=big_shapes_non_singleton)


class TestTensorShapeManipulation(unittest.TestCase):
    def test_basic_slicing(self):
        tensor_np = np.random.randint(0, 10, size=(3, 4, 3, 1, 2))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(tensor_np[0].tolist(), tensor_ctrlc[0].tolist(), f"Tensor getitem does not work.")
        self.assertEqual(tensor_np[1:2].tolist(), tensor_ctrlc[1:2].tolist(), f"Tensor getslice does not work.")
        self.assertEqual(tensor_np[:2].tolist(), tensor_ctrlc[:2].tolist(), f"Tensor getslice does not work.")
        self.assertEqual(tensor_np[:].tolist(), tensor_ctrlc[:].tolist(), f"Tensor getslice does not work.")
        self.assertEqual(tensor_np[2:].tolist(), tensor_ctrlc[2:].tolist(), f"Tensor getslice does not work.")
        self.assertEqual(tensor_np[0, 0].tolist(), tensor_ctrlc[0, 0].tolist(), f"Tensor multidim getitem does not work.")
        self.assertEqual(tensor_np[0, 2:3].tolist(), tensor_ctrlc[0, 2:3].tolist(), f"Tensor multidim getitem does not work.")
        self.assertEqual(tensor_np[:, 2].tolist(), tensor_ctrlc[:, 2].tolist(),f"Tensor multidim getitem does not work.")

        self.assertEqual(tensor_np[0, 2:3, :, 0].tolist(), tensor_ctrlc[0, 2:3, :, 0].tolist(), f"Tensor multidim getitem does not work.")
        self.assertEqual(tensor_np[:, 2, :3, 0].tolist(), tensor_ctrlc[:, 2, :3, 0].tolist(), f"Tensor multidim getitem does not work.")

    def test_basic_flatten(self):
        tensor_np = np.random.randint(0, 10, size=(3, 4, 3, 1, 2))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(tensor_np[0, 0, 0, 0].flatten().tolist(), tensor_ctrlc[0, 0, 0, 0].flatten().tolist(), f"Tensor flatten does not work.")
        self.assertEqual(tensor_np[0, 0, 0].flatten().tolist(), tensor_ctrlc[0, 0, 0].flatten().tolist(), f"Tensor flatten does not work.")
        self.assertEqual(tensor_np[0, 0].flatten().tolist(), tensor_ctrlc[0, 0].flatten().tolist(), f"Tensor flatten does not work.")
        self.assertEqual(tensor_np.flatten().tolist(), tensor_ctrlc.flatten().tolist(), f"Tensor flatten does not work.")


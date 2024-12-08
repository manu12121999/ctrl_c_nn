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
                try:
                    result_A = func1(mat1, mat2).tolist()
                except (ValueError, AssertionError, RuntimeError):
                    result_A = None
                try:
                    result_B = func2(mat1_B, mat2_B).tolist()
                except (ValueError, AssertionError, RuntimeError):
                    result_B = None

                self.assertEqual(result_A, result_B,
                                 f"Tensor operation {func2} failed for shapes {mat1.shape} and {mat2.shape}. \n "
                                 + f"Input matrices \n {mat1} \n and \n {mat2}.\n "
                                 +  f"Results are numpy: \n {result_A} \n and ctrl_c_nn \n {result_B} ")

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

class TestTensorUnaryAndReductions(unittest.TestCase):

    def test_abs(self):
        mat1 = np.random.randint(-10, 10, size=(1, 4, 6, 2, 1, 3))
        result_A = np.abs(mat1)
        result_B = ccm.Tensor(mat1.tolist()).abs()
        self.assertEqual(result_A.tolist(), result_B.tolist(), f"Tensor operation abs failed.")

    def test_sum(self):
        mat1 = np.random.randint(-10, 10, size=(1, 2, 4, 6, 1, 3, 1))
        self.assertEqual(mat1.sum(0).tolist(), ccm.Tensor(mat1.tolist()).sum(0).tolist(), f"Tensor sum(0) failed.")
        self.assertEqual(mat1.sum(1).tolist(), ccm.Tensor(mat1.tolist()).sum(1).tolist(), f"Tensor sum(1) failed.")
        self.assertEqual(mat1.sum(2).tolist(), ccm.Tensor(mat1.tolist()).sum(2).tolist(), f"Tensor sum(2) failed.")
        self.assertEqual(mat1.sum(3).tolist(), ccm.Tensor(mat1.tolist()).sum(3).tolist(), f"Tensor sum(3) failed.")
        self.assertEqual(mat1.sum(4).tolist(), ccm.Tensor(mat1.tolist()).sum(4).tolist(), f"Tensor sum(4) failed.")
        self.assertEqual(mat1.sum(5).tolist(), ccm.Tensor(mat1.tolist()).sum(5).tolist(), f"Tensor sum(5) failed.")
        self.assertEqual(mat1.sum(6).tolist(), ccm.Tensor(mat1.tolist()).sum(6).tolist(), f"Tensor sum(6) failed.")

    def test_prod(self):
        mat1 = np.random.randint(-10, 10, size=(1, 2, 4, 6, 1, 3, 1))
        self.assertEqual(mat1.prod(0).tolist(), ccm.Tensor(mat1.tolist()).prod(0).tolist(), f"Tensor prod(0) failed.")
        self.assertEqual(mat1.prod(1).tolist(), ccm.Tensor(mat1.tolist()).prod(1).tolist(), f"Tensor prod(1) failed.")
        self.assertEqual(mat1.prod(2).tolist(), ccm.Tensor(mat1.tolist()).prod(2).tolist(), f"Tensor prod(2) failed.")
        self.assertEqual(mat1.prod(3).tolist(), ccm.Tensor(mat1.tolist()).prod(3).tolist(), f"Tensor prod(3) failed.")
        self.assertEqual(mat1.prod(4).tolist(), ccm.Tensor(mat1.tolist()).prod(4).tolist(), f"Tensor prod(4) failed.")
        self.assertEqual(mat1.prod(5).tolist(), ccm.Tensor(mat1.tolist()).prod(5).tolist(), f"Tensor prod(5) failed.")
        self.assertEqual(mat1.prod(6).tolist(), ccm.Tensor(mat1.tolist()).prod(6).tolist(), f"Tensor prod(6) failed.")


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

    def test_basic_unsqueeze(self):
        tensor_np = np.random.randint(0, 10, size=(3, 4, 3, 1, 2))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(np.expand_dims(tensor_np, 0).tolist(), tensor_ctrlc.unsqueeze(0).tolist(), f"Tensor unsqueeze dim=0 does not work.")
        self.assertEqual(np.expand_dims(tensor_np, 1).tolist(), tensor_ctrlc.unsqueeze(1).tolist(), f"Tensor unsqueeze dim=1 does not work.")
        self.assertEqual(np.expand_dims(tensor_np, 2).tolist(), tensor_ctrlc.unsqueeze(2).tolist(), f"Tensor unsqueeze dim=2 does not work.")
        self.assertEqual(np.expand_dims(tensor_np, 4).tolist(), tensor_ctrlc.unsqueeze(4).tolist(), f"Tensor unsqueeze dim=4 does not work.")
        self.assertEqual(np.expand_dims(tensor_np, 5).tolist(), tensor_ctrlc.unsqueeze(5).tolist(), f"Tensor unsqueeze dim=5 does not work.")

    def test_basic_squeeze(self):
        tensor_np = np.random.randint(0, 10, size=(1, 4, 3, 1, 2, 1))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(np.squeeze(tensor_np, 0).tolist(), tensor_ctrlc.squeeze(0).tolist(), f"Tensor unsqueeze dim=0 does not work.")
        self.assertEqual(np.squeeze(tensor_np, 3).tolist(), tensor_ctrlc.squeeze(3).tolist(), f"Tensor unsqueeze dim=3 does not work.")
        self.assertEqual(np.squeeze(tensor_np, 5).tolist(), tensor_ctrlc.squeeze(5).tolist(), f"Tensor unsqueeze dim=5 does not work.")

    def test_reshape(self):
        tensor_np = np.random.randint(0, 10, size=(5,3))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(tensor_np.reshape([3,5]).tolist(), tensor_ctrlc.reshape([3,5]).tolist(), f"Tensor reshape 5,3 to 3,5 does not work.")
        self.assertEqual(tensor_np.reshape([15]).tolist(), tensor_ctrlc.reshape([15]).tolist(), f"Tensor reshape 5,3 to 15 does not work.")
        self.assertEqual(tensor_np.reshape([3,5,1]).tolist(), tensor_ctrlc.reshape([3,5,1]).tolist(), f"Tensor reshape 5,3 to 3,5,1 does not work.")
        self.assertEqual(tensor_np.reshape([1,1,3,5,1]).tolist(), tensor_ctrlc.reshape([1,1,3,5,1]).tolist(), f"Tensor reshape 5,3 to 3,5,1 does not work.")

    def test_setitem(self):
        tensor_np = np.zeros((5, 10))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())

        tensor_np[(3, 2)] = 1
        tensor_ctrlc[(3, 2)] = 1
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem scalar does not work.")

        tensor_np[1, 5] = -1
        tensor_ctrlc[1, 5] = -1
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem scalar does not work.")

        tensor_np[1] = np.ones((10))
        tensor_ctrlc[1] = ccm.Tensor.ones((10))
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem array does not work.")

        tensor_np[0, 3:5] = np.ones((2,))
        tensor_ctrlc[0, 3:5] = ccm.Tensor.ones((2,))
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem array does not work.")

    def test_permute(self):
        tensor_np = np.random.randint(0, 10, size=(5, 3))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(np.transpose(tensor_np, (1, 0)).tolist(), tensor_ctrlc.permute((1, 0)).tolist(), f"Tensor permute 2d does not work.")

        tensor_np = np.random.randint(0, 10, size=(5, 3, 3))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(np.transpose(tensor_np, (1, 0, 2)).tolist(), tensor_ctrlc.permute((1, 0, 2)).tolist(), f"Tensor permute 3d does not work.")
        self.assertEqual(np.transpose(tensor_np, (0, 2, 1)).tolist(), tensor_ctrlc.permute((0, 2, 1)).tolist(), f"Tensor permute 3d does not work.")
        self.assertEqual(np.transpose(tensor_np, (0, 1, 2)).tolist(), tensor_ctrlc.permute((0, 1, 2)).tolist(), f"Tensor permute 3d does not work.")

        tensor_np = np.random.randint(0, 10, size=(5, 3, 3, 3))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(np.transpose(tensor_np, (1, 0, 3, 2)).tolist(), tensor_ctrlc.permute((1, 0, 3, 2)).tolist(), f"Tensor permute 4d does not work.")
        self.assertEqual(np.transpose(tensor_np, (3, 2, 1, 0)).tolist(), tensor_ctrlc.permute((3, 2, 1, 0)).tolist(), f"Tensor permute 4d does not work.")
        self.assertEqual(np.transpose(tensor_np, (0, 3, 2, 1)).tolist(), tensor_ctrlc.permute((0, 3, 2, 1)).tolist(), f"Tensor permute 4d does not work.")

        tensor_np = np.random.randint(0, 10, size=(1, 5, 3, 3, 3, 1))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(np.transpose(tensor_np, (4, 1, 0, 3, 5, 2)).tolist(), tensor_ctrlc.permute((4, 1, 0, 3, 5, 2)).tolist(), f"Tensor permute 6d does not work.")


import unittest
import numpy as np

from ctrl_c_nn import Tensor

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

    def test_mul_dimless_tensor(self):
        mat1 = np.array([3, 2])
        mat2 = np.array(2)
        result_A = mat1 * mat2
        result_B = ccm.Tensor(mat1.tolist()) * ccm.Tensor(mat2.tolist())
        self.assertEqual(result_A.tolist(), result_B.tolist(), f"Tensor operation dimless mul failed.")
        result_C = mat2 * mat1
        result_D = ccm.Tensor(mat2.tolist()) * ccm.Tensor(mat1.tolist())
        self.assertEqual(result_C.tolist(), result_D.tolist(), f"Tensor operation dimless mul failed.")

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

        def test_slicing_assignment(self):
        tensor_np = np.random.randint(0, 10, size=(3, 4, 3, 1, 2))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())

        tensor_np[0] = tensor_np[1]
        tensor_ctrlc[0] = tensor_ctrlc[1]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem does not work (A).")
        tensor_np[1] = tensor_np[2]
        tensor_ctrlc[1] = tensor_ctrlc[2]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem does not work (A2).")
        tensor_np[1:2] = tensor_np[2:3]
        tensor_ctrlc[1:2] = tensor_ctrlc[2:3]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (B).")
        tensor_np[:2] = tensor_np[1:3]
        tensor_ctrlc[:2] = tensor_ctrlc[1:3]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (C).")
        tensor_np[:] = tensor_np[:]
        tensor_ctrlc[:] = tensor_ctrlc[:]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (D).")
        tensor_np[2:] = tensor_np[1:-1]
        tensor_ctrlc[2:] = tensor_ctrlc[1:-1]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (E).")
        tensor_np[2, 3] = tensor_np[2, 1]
        tensor_ctrlc[2, 3] = tensor_ctrlc[2, 1]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem multidim does not work (F).")
        tensor_np[2, 1] = tensor_np[2, 0]
        tensor_ctrlc[2, 1] = tensor_ctrlc[2, 0]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setitem multidim not work (F2).")
        tensor_np[0, 2:3] = tensor_np[0, 1:2]
        tensor_ctrlc[0, 2:3] = tensor_ctrlc[0, 1:2]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (G).")
        tensor_np[:, 2] = tensor_np[:, 1]
        tensor_ctrlc[:, 2] = tensor_ctrlc[:, 1]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (H).")
        tensor_np[0, 2:3, :, 0] = tensor_np[0, 3:4, :, 0]
        tensor_ctrlc[0, 2:3, :, 0] = tensor_ctrlc[0, 3:4, :, 0]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (I).")
        tensor_np[:, 2, :3, 0] = tensor_np[:, 2, :3, 0]
        tensor_ctrlc[:, 2, :3, 0] = tensor_ctrlc[:, 2, :3, 0]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (J).")
        tensor_np[:, 2, :3] = tensor_np[:, 2, :3]
        tensor_ctrlc[:, 2, :3] = tensor_ctrlc[:, 2, :3]
        self.assertEqual(tensor_np.tolist(), tensor_ctrlc.tolist(), f"Tensor setslice does not work (J).")

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

    def test_basic_sum(self):
        tensor_np = np.random.randint(0, 10, size=(3, 4, 3, 1, 2))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(tensor_np.sum().tolist(), tensor_ctrlc.sum().tolist(), f"Tensor sum all does not work.")
        self.assertEqual(tensor_np.sum(1).tolist(), tensor_ctrlc.sum(1).tolist(), f"Tensor sum all does not work.")
        self.assertEqual(tensor_np.sum(4).tolist(), tensor_ctrlc.sum(4).tolist(), f"Tensor sum all does not work.")
        self.assertEqual(tensor_np.sum((0,)).tolist(), tensor_ctrlc.sum(0,).tolist(), f"Tensor sum all does not work.")
        self.assertEqual(tensor_np.sum((3,)).tolist(), tensor_ctrlc.sum(3,).tolist(), f"Tensor sum all does not work.")
        self.assertEqual(tensor_np.sum((4,)).tolist(), tensor_ctrlc.sum(4, ).tolist(), f"Tensor sum all does not work.")
        self.assertEqual(tensor_np.sum((1, 3)).tolist(), tensor_ctrlc.sum((1, 3)).tolist(), f"Tensor sum all does not work.")
        self.assertEqual(tensor_np.sum((0, 4)).tolist(), tensor_ctrlc.sum((0, 4)).tolist(), f"Tensor sum all does not work.")

    def test_basic_mean(self):
        tensor_np = np.ones((3, 4, 3, 1, 2))  # TODO: other values than 1
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(tensor_np.mean().tolist(), tensor_ctrlc.mean().tolist(), f"Tensor mean all does not work.")
        self.assertListEqual(tensor_np.mean(1).tolist(), tensor_ctrlc.mean(1).tolist(), f"Tensor mean all does not work.")
        self.assertListEqual(tensor_np.mean(4).tolist(), tensor_ctrlc.mean(4).tolist(), f"Tensor mean all does not work.")
        self.assertListEqual(tensor_np.mean((0,)).tolist(), tensor_ctrlc.mean(0, ).tolist(), f"Tensor mean all does not work.")
        self.assertListEqual(tensor_np.mean((3,)).tolist(), tensor_ctrlc.mean(3, ).tolist(), f"Tensor mean all does not work.")
        self.assertListEqual(tensor_np.mean((4,)).tolist(), tensor_ctrlc.mean(4, ).tolist(), f"Tensor mean all does not work.")
        self.assertListEqual(tensor_np.mean((1, 3)).tolist(), tensor_ctrlc.mean((1, 3)).tolist(), f"Tensor mean all does not work.")
        self.assertListEqual(tensor_np.mean((0, 4)).tolist(), tensor_ctrlc.mean((0, 4)).tolist(), f"Tensor mean all does not work.")

    def test_basic_max(self):
        tensor_np = np.random.randint(0, 10, size=(3, 4, 3, 1, 2))
        tensor_ctrlc = ccm.Tensor(tensor_np.tolist())
        self.assertEqual(tensor_np.max().tolist(), tensor_ctrlc.max().tolist(), f"Tensor max all does not work.")
        self.assertListEqual(tensor_np.max(1).tolist(), tensor_ctrlc.max(1).tolist(), f"Tensor max all does not work.")
        self.assertListEqual(tensor_np.max(4).tolist(), tensor_ctrlc.max(4).tolist(), f"Tensor max all does not work.")
        self.assertListEqual(tensor_np.max((0,)).tolist(), tensor_ctrlc.max(0, ).tolist(), f"Tensor max all does not work.")
        self.assertListEqual(tensor_np.max((3,)).tolist(), tensor_ctrlc.max(3, ).tolist(), f"Tensor max all does not work.")
        self.assertListEqual(tensor_np.max((4,)).tolist(), tensor_ctrlc.max(4, ).tolist(), f"Tensor max all does not work.")
        self.assertListEqual(tensor_np.max((1, 3)).tolist(), tensor_ctrlc.max((1, 3)).tolist(), f"Tensor max all does not work.")
        self.assertListEqual(tensor_np.max((0, 4)).tolist(), tensor_ctrlc.max((0, 4)).tolist(), f"Tensor max all does not work.")

class TestTensorCreation(unittest.TestCase):

    def test_create(self):
        l1 = [[2, 1, 3], [4, 2, 1]]
        self.assertEqual(l1, Tensor(l1).tolist(), f"Tensor create and to_list is not identical.")
        shape = (3, 2, 1)
        t1 = Tensor.zeros(shape)
        t2 = Tensor.ones(shape)
        t3 = Tensor.random_float(shape)
        t4 = Tensor.random_int(shape)
        t5 = Tensor.random_float(shape, min=-2, max=+2)
        t6 = Tensor.random_int(shape, min=-2, max=+2)
        t7 = Tensor.fill(shape, 27)
        self.assertEqual(shape, t1.shape, f"Tensor create gives wrong shape.")
        self.assertEqual(shape, t2.shape, f"Tensor create gives wrong shape.")
        self.assertEqual(shape, t3.shape, f"Tensor create gives wrong shape.")
        self.assertEqual(shape, t4.shape, f"Tensor create gives wrong shape.")
        self.assertEqual(shape, t5.shape, f"Tensor create gives wrong shape.")
        self.assertEqual(shape, t6.shape, f"Tensor create gives wrong shape.")
        self.assertEqual(shape, t7.shape, f"Tensor create gives wrong shape.")

    def test_create_single_int(self):
        n1 = np.array(4)
        t1 = Tensor(4)
        self.assertEqual(n1.shape, t1.shape, f"Wrong shape when create tensor from int.")
        self.assertEqual(n1.ndim, t1.ndim, f"Wrong ndim when create tensor from int.")

        self.assertEqual((n1 * n1).shape, (t1 * t1).shape, f"Wrong shape when create tensor from int.")
        self.assertEqual((n1 * n1).ndim, (t1 * t1).ndim, f"Wrong ndim when create tensor from int.")

        n2 = np.array([8, 3, 4])
        t2 = Tensor([8, 3, 4])

        self.assertEqual((n1*n2).shape, (t1*t2).shape, f"Wrong shape when create tensor from int.")
        self.assertEqual((n1*n2).ndim, (t1*t2).ndim, f"Wrong ndim when create tensor from int.")

        self.assertEqual((n2*n1).shape, (t2*t1).shape, f"Wrong shape when create tensor from int.")
        self.assertEqual((n2*n1).ndim, (t2*t1).ndim, f"Wrong ndim when create tensor from int.")

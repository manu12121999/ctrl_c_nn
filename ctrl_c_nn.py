import sys
import math
import operator
from typing import List, Tuple, Any

sumprod = math.sumprod if sys.version_info >= (3, 12) else lambda p, q: sum([p_i*q_i for p_i, q_i in zip(p, q)])


class LLOps:
    """
    Class for (recursive) functional operations on lists of lists
    """
    @staticmethod
    def fill(shape: Tuple, value):
        # Make a new list (of lists) filled with value
        if len(shape) == 1:
            return [value for _ in range(shape[0])]
        else:
            return [LLOps.fill(shape[1:], value) for _ in range(shape[0])]

    @staticmethod
    def f_operator_scalar(a: List, b: (int, float), op):
        # Add a scalar to a list (of lists). Other operations than add are supported too
        if isinstance(a[0], (int, float)):
            return [op(a_i, b) for a_i in a]
        else:
            return [LLOps.f_operator_scalar(a_i, b, op) for a_i in a]

    @staticmethod
    def f_operator_same_size(a: List, b: List, op):
        # Add two list (of lists). Other operations than add are supported too
        if isinstance(a[0], (int, float)):
            return [op(a_i, b_i) for a_i, b_i in zip(a, b)]
        else:
            return [LLOps.f_operator_same_size(a_i, b_i, op) for a_i, b_i in zip(a, b)]

    @staticmethod
    def f_transpose_2dim(a: List):
        # Transpose a 2-dimensional list
        # (I,J) -> (J,I)
        I, J = len(a), len(a[0])
        return [[a[i][j] for i in range(I)] for j in range(J)]

    @staticmethod
    def f_matmul_2d(a: List, b: List):
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        b_T = LLOps.f_transpose_2dim(b)
        return [[sumprod(a[i], b_T[j]) for j in range(J)] for i in range(I)]

    @staticmethod
    def f_matmul_2d_old(a: List, b: List):
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        result = [[0 for _ in range(J)] for _ in range(I)]

        # Transpose
        b_T = LLOps.f_transpose_2dim(b)

        # Perform matrix multiplication
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    result[i][j] += a[i][k] * b_T[j][k]

        return result

    @staticmethod
    def f_vec_times_vec(a: List, b: List):
        # perform vector times matrix multiplication on a 1-dimensional list and another 2-dimensional lists
        assert len(a) == len(b)
        return sumprod(a, b)

    @staticmethod
    def f_mat_times_vec(a: List, b: List):
        # perform matrix times vector multiplication on a 2-dimensional list and another 1-dimensional lists
        return [LLOps.f_vec_times_vec(row, b) for row in a]

    @staticmethod
    def f_vec_times_mat(a: List, b: List):
        # perform vector times vector multiplication on two 1-dimensional lists
        return [LLOps.f_vec_times_vec(a, row) for row in LLOps.f_transpose_2dim(b)]

    @staticmethod
    def f_squeeze(a: List, dim):
        # remove one dimension from a list of lists
        if dim == 0:
            return a[0]
        elif dim == 1:
            return [a_i[0] for a_i in a]
        else:
            return [LLOps.f_squeeze(a_i, dim-1) for a_i in a]

    @staticmethod
    def f_slice(a: List, item: Tuple):
        # Return a slice of the list of lists (e.g. a[0] or a[:, 3:2])
        if len(item) == 1:
            return a[item[0]]
        else:
            index = item[0]
            if isinstance(index, slice):
                return [LLOps.f_slice(sublist, item[1:]) for sublist in a[index]]
            else:  # isinstance(index, int):
                return LLOps.f_slice(a[index], item[1:])

    @staticmethod
    def f_flatten(a: List):
        # Flatten a list of lists into a single list
        if isinstance(a[0], list):
            if isinstance(a[0][0], list):
                return [subsublist for sublist in a for subsublist in LLOps.f_flatten(sublist)]
            else:
                return [num for sublist in a for num in sublist]
        else:
            return a


class Tensor:
    # Wrapper to use linalg operations on lists (of lists) (e.g. matmuls) in a nicer way

    def __init__(self, elems):
        self.elems = elems

        # calculate number of dimensions
        self.ndim = 0
        self.shape = []
        self._calc_shape_and_dims()

    def _calc_shape_and_dims(self):
        shape_list = []
        a = self.elems
        while isinstance(a, list) and len(a) > 0:
            shape_list.append(len(a))
            a = a[0]
            self.ndim += 1
        self.shape = tuple(shape_list)

    def __repr__(self):
        return f"Tensor of shape {self.shape}.  Elements ({self.elems})"

    @staticmethod
    def zeros(shape):
        return Tensor(LLOps.fill(shape, 0))

    @staticmethod
    def ones(shape):
        return Tensor(LLOps.fill(shape, 1))

    @staticmethod
    def fill(shape, number):
        return Tensor(LLOps.fill(shape, number))

    @staticmethod
    def _basic_op(a, b, op):
        # input tensor output list of list
        if isinstance(b, Tensor):
            # print(f"add/mul shapes {a.shape} and {b.shape}")
            if a.shape == b.shape:
                return LLOps.f_operator_same_size(a.elems, b.elems, op)
            elif a.ndim == b.ndim:
                if a.shape[0] == 1:
                    if a.ndim == 1 and b.ndim == 1:
                        return [op(a.elems[0], b_i) for b_i in b.elems]
                    else:
                        return [Tensor._basic_op(Tensor(a.elems[0]), Tensor(b_i), op) for b_i in b.elems]
                elif b.shape[0] == 1:
                    if a.ndim == 1 and b.ndim == 1:
                        return [op(a_i, b.elems[0]) for a_i in a.elems]
                    else:
                        return [Tensor._basic_op(Tensor(a_i), Tensor(b.elems[0]), op) for a_i in a.elems]
                elif a.shape[0] == b.shape[0]:
                    return [Tensor._basic_op(Tensor(a_i), Tensor(b_i), op) for a_i, b_i in zip(a.elems, b.elems)]
                else:
                    raise AssertionError()
            elif a.ndim < b.ndim:
                return [Tensor._basic_op(a, Tensor(b_i), op) for b_i in b.elems]
            elif a.ndim > b.ndim:
                return [Tensor._basic_op(Tensor(a_i), b, op) for a_i in a.elems]
            else:
                raise AssertionError()
        elif isinstance(b, (float, int)):
            return LLOps.f_operator_scalar(a.elems, b, op)
        else:
            raise NotImplementedError("type", type(b))

    def __add__(self, other):
        return Tensor(Tensor._basic_op(self, other, operator.add))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Tensor(Tensor._basic_op(self, other, operator.sub))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return Tensor(Tensor._basic_op(self, other, operator.mul))

    def __rmul__(self, other):
        return self.__mul__(other)

    @staticmethod
    def _f_matmul(a, b):
        # input types: Tensors, output type: list
        # print(f"multiplying shapes {a.shape} and {b.shape}")
        if a.ndim == 2 and b.ndim == 2:
            return LLOps.f_matmul_2d(a.elems, b.elems)
        # elif a.ndim == 2 and b.ndim == 3:
        #     print("dims", a.ndim, b.ndim)
        #     return [LLOps.f_matmul_2dim(a.elems, b_i) for b_i in b.elems]
        # elif a.ndim == 3 and b.ndim == 2:
        #     print("dims", a.ndim, b.ndim)
        #     return [LLOps.f_matmul_2dim(a_i, b.elems) for a_i in a.elems]
        elif a.ndim == 2 and b.ndim >= 3:
            return [Tensor._f_matmul(a, Tensor(b_i)) for b_i in b.elems]
        elif a.ndim >= 3 and b.ndim == 2:
            return [Tensor._f_matmul(Tensor(a_i), b) for a_i in a.elems]
        elif a.ndim >= 3 and b.ndim >= 3:
            if a.ndim == b.ndim:
                if len(a.elems) == len(b.elems):
                    return [Tensor._f_matmul(Tensor(a_i), Tensor(b_i)) for a_i, b_i in zip(a.elems, b.elems)]
                elif len(a.elems) == 1:
                    return [Tensor._f_matmul(Tensor(a.elems[0]), Tensor(b_i)) for b_i in b.elems]
                elif len(b.elems) == 1:
                    return [Tensor._f_matmul(Tensor(a_i), Tensor(b.elems[0])) for a_i in a.elems]
                else:
                    raise AssertionError()
            elif a.ndim > b.ndim:
                return [Tensor._f_matmul(Tensor(a_i), b) for a_i in a.elems]
            else:  # a.ndim < b.ndim:
                return [Tensor._f_matmul(a, Tensor(b_i)) for b_i in b.elems]
        elif a.ndim == 1 and b.ndim == 1:
            return LLOps.f_vec_times_vec(a.elems, b.elems)
        elif a.ndim == 2 and b.ndim == 1:
            return LLOps.f_mat_times_vec(a.elems, b.elems)
        elif a.ndim == 1 and b.ndim == 2:
            return LLOps.f_vec_times_mat(a.elems, b.elems)
        elif a.ndim == 1 and b.ndim > 2:
            return [Tensor._f_matmul(a, Tensor(b_i)) for b_i in b.elems]
        elif a.ndim > 2 and b.ndim == 1:
            return [Tensor._f_matmul(Tensor(a_i), b) for a_i in a.elems]
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        assert isinstance(other, Tensor) is True
        return Tensor(Tensor._f_matmul(self, other))

    def inv(self):
        raise NotImplementedError

    def T(self):
        if self.ndim == 2:
            return Tensor(LLOps.f_transpose_2dim(self.elems))
        else:
            raise NotImplementedError

    def squeeze(self, dim):
        return LLOps.f_squeeze(self.elems, dim)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return Tensor(self.elems[item])
        elif isinstance(item, tuple):
            return Tensor(LLOps.f_slice(self.elems, item))

    def flatten(self):
        return Tensor(LLOps.f_flatten(self.elems))

    def tolist(self):
        return self.elems


if __name__ == "__main__":
    a = Tensor.zeros((4, 8, 2))
    b = Tensor.zeros((2, 8))
    print("Result matmul \n ", a@b)

    print("Result slice", a[0, 3:4, :])
    c = a[:2, :, 0]
    print("Result add \n ", a[:2, :, 0] + b)

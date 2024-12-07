import sys
import math
import operator
from functools import partial
from typing import List, Tuple, Any
from multiprocessing import Pool

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
    def f_unary_op(a: List, f):
        # input tensor output list of list
        if isinstance(a[0], list):
            return [LLOps.f_unary_op(a_i, f) for a_i in a]
        else:
            return [f(a_i) for a_i in a]

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
    def f_transpose_2d(a: List):
        # Transpose a 2-dimensional list
        # (I,J) -> (J,I)
        I, J = len(a), len(a[0])
        return [[a[i][j] for i in range(I)] for j in range(J)]

    @staticmethod
    def f_clone_2d(a: List):
        # Deep-copy a 2-dimensional list of shape (I, J)
        I, J = len(a), len(a[0])
        return [[a[i][j] for j in range(J)] for i in range(I)]

    @staticmethod
    def f_matmul_2d(a: List, b: List):
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        b_T = LLOps.f_transpose_2d(b)
        return [[sumprod(a[i], b_T[j]) for j in range(J)] for i in range(I)]

    @staticmethod
    def f_matmul_2d_multiprocess(a: List, b: List):
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        with Pool(8) as p:
            # return p.map(partial(LLOps.f_vec_times_mat, b=b), a)
            # return p.map(LLOps.f_vec_times_mat_c, ((a_i, LLOps.f_clone_2d(b)) for a_i in a))
            return p.starmap(LLOps.f_vec_times_mat, ((a_i, b) for a_i in a), chunksize=max(1, I//8))

    @staticmethod
    def f_matmul_2d_old(a: List, b: List):
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        result = [[0 for _ in range(J)] for _ in range(I)]

        # Transpose
        b_T = LLOps.f_transpose_2d(b)

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
        return [LLOps.f_vec_times_vec(a, row) for row in LLOps.f_transpose_2d(b)]

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
    def f_unsqueeze(a: List, dim):
        # remove one dimension from a list of lists
        if dim == 0:
            return [a]
        elif dim == 1:
            return [[a_i] for a_i in a]
        else:
            return [LLOps.f_unsqueeze(a_i, dim-1) for a_i in a]

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

    @staticmethod
    def f_setitem(a: List, key: Tuple[int], value):
        # set the item at position key of list a to a value. Value can be scalar or list.  (a[key] = value)
        if len(key) == 1:
            a[key[0]] = value
        else:
            LLOps.f_setitem(a[key[0]], key[1:], value)

    @staticmethod
    def f_reshape_flattened(a: List[(int | float)], shape: Tuple[int]):
        # reshape a one-dimensional array (flattened) into a target format
        if len(shape) == 1:
            return a
        else:
            n = len(a) // shape[0]  # 2
            return [LLOps.f_reshape_flattened(a[i*n:(i+1)*n], shape[1:]) for i in range(shape[0])]


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

    ######################
    # Construction Methods
    #####################

    @staticmethod
    def zeros(shape):
        if isinstance(shape, int):
            shape = (shape, )
        return Tensor(LLOps.fill(shape, 0))

    @staticmethod
    def ones(shape):
        if isinstance(shape, int):
            shape = (shape, )
        return Tensor(LLOps.fill(shape, 1))

    @staticmethod
    def fill(shape, number):
        if isinstance(shape, int):
            shape = (shape, )
        return Tensor(LLOps.fill(shape, number))

    ########################
    # Arithmetic Operations
    #######################

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
        #     return [LLOps.f_matmul_2d(a.elems, b_i) for b_i in b.elems]
        # elif a.ndim == 3 and b.ndim == 2:
        #     print("dims", a.ndim, b.ndim)
        #     return [LLOps.f_matmul_2d(a_i, b.elems) for a_i in a.elems]
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
            return Tensor(LLOps.f_transpose_2d(self.elems))
        else:
            raise NotImplementedError

    def abs(self):
        return Tensor(LLOps.f_unary_op(self.elems, abs))

    ######################
    # Shape Manipulation
    #####################

    def squeeze(self, dim):
        return Tensor(LLOps.f_squeeze(self.elems, dim))

    def unsqueeze(self, dim):
        return Tensor(LLOps.f_unsqueeze(self.elems, dim))

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return Tensor(self.elems[item])
        elif isinstance(item, tuple):
            return Tensor(LLOps.f_slice(self.elems, item))

    def __setitem__(self, key, value):
        v = value.elems if isinstance(value, Tensor) else value
        if isinstance(key, (int, slice)):
            self.elems[key] = v
        else:  # key is tuple, list, or other iterable
            LLOps.f_setitem(self.elems, key, v)

    def flatten(self):
        return Tensor(LLOps.f_flatten(self.elems))

    def tolist(self):
        return self.elems

    def item(self):
        return self.elems

    def reshape(self, shape):
        return Tensor(LLOps.f_reshape_flattened(LLOps.f_flatten(self.elems), shape))

    def reshape_old(self, shape):
        def calc_card(a):
            m = 1
            card = []
            mod = []
            for n in reversed(a):
                card.insert(0, m)
                m *= n
                mod.insert(0, m)
            return card, mod

        flat_tensor = self.flatten()
        new_tensor = Tensor.zeros(shape)
        for i in range(flat_tensor.shape[0]):
            card, mod = calc_card(shape)
            index = [(i//c)%m for c,m in zip(card, mod)]
            new_tensor[index] = flat_tensor[i].item()
        return new_tensor


if __name__ == "__main__":
    a_list = [[2, 3],
              [32, -21],
              [32, 21]]  # 3,2

    a_reshaped = Tensor(a_list).reshape((2, 3))
    print(a_reshaped.shape)
    print(a_reshaped.abs())

    b_list = [[4, 2],
              [12, 22]]          # 2,2
    print("matmul of lists", LLOps.f_matmul_2d(a_list, b_list))
    print("matmul of lists (multiproc)", LLOps.f_matmul_2d_multiprocess(a_list, b_list))

    a_tensor = Tensor.zeros((2, 4, 8, 2))
    b_tensor = Tensor.zeros((2, 8))
    c_tensor = a_tensor@b_tensor  # shape (2, 4, 8, 8)
    d_tensor = c_tensor[0, 2:, :, :1] + b_tensor.unsqueeze(2)  # shape (2,8,1)
    print(d_tensor)

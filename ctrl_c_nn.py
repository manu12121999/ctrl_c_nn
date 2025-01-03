__author__ = "Manuel Vogel"
__version__ = "0.0.2"
__website__ = "https://github.com/manu12121999/ctrl_c_nn"
__original_source__ = "https://github.com/manu12121999/ctrl_c_nn/blob/main/ctrl_c_nn.py"
__email__ = "manu12121999@gmail.com"

import collections
import copy
import pickle
import random
import struct
import time
import zlib
import sys
import math
import operator
from multiprocessing import Pool
from zipfile import ZipFile

sumprod = math.sumprod if sys.version_info >= (3, 12) else lambda p, q: sum([p_i*q_i for p_i, q_i in zip(p, q)])


class LLOps:
    """
    Class for (recursive) functional operations on lists of lists
    """

    @staticmethod
    def fill(shape: tuple, value: (int, float)) -> list:
        # Make a new list (of lists) filled with value
        if len(shape) == 1:
            return [value for _ in range(shape[0])]
        else:
            return [LLOps.fill(shape[1:], value) for _ in range(shape[0])]

    @staticmethod
    def fill_callable(shape: tuple, gen: callable) -> list:
        # Make a new list (of lists) filled with values generated from the callable gen
        if len(shape) == 1:
            return [gen() for _ in range(shape[0])]
        else:
            return [LLOps.fill(shape[1:], gen()) for _ in range(shape[0])]

    @staticmethod
    def f_unary_op(a: list, f: callable) -> list:
        # Apply function f to all elements
        if not isinstance(a, list):
            return f(a)
        if not isinstance(a[0], list):
            return [f(a_i) for a_i in a]
        else:
            return [LLOps.f_unary_op(a_i, f) for a_i in a]

    @staticmethod
    def f_binary_op_scalar(a: list, b: (int, float), op: callable) -> list:
        # Add a scalar to a list (of lists). Other operations than add are supported too
        if isinstance(a, (int, float)):
            return op(a, b)
        elif isinstance(a[0], (int, float)):
            return [op(a_i, b) for a_i in a]
        else:
            return [LLOps.f_binary_op_scalar(a_i, b, op) for a_i in a]

    @staticmethod
    def f_binary_op_same_size(a: list, b: list, op: callable) -> list:
        # Add two list (of lists). Other operations than add are supported too
        if isinstance(a[0], (int, float)):
            return [op(a_i, b_i) for a_i, b_i in zip(a, b)]
        else:
            return [LLOps.f_binary_op_same_size(a_i, b_i, op) for a_i, b_i in zip(a, b)]

    @staticmethod
    def f_add_same_size_performance(a: list, b: list) -> list:
        # Add two list (of lists). Only used to test the performance of different implementations
        if isinstance(a[0], (int, float)):
            return [a_i + b_i for a_i, b_i in zip(a, b)]
        else:
            return [LLOps.f_add_same_size_performance(a_i, b_i) for a_i, b_i in zip(a, b)]

    @staticmethod
    def f_transpose_2d(a: list) -> list:
        # Transpose a 2-dimensional list
        # (I,J) -> (J,I)
        I, J = len(a), len(a[0])
        return [[a[i][j] for i in range(I)] for j in range(J)]

    @staticmethod
    def f_matmul_2d(a: list, b: list) -> list:
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        b_T = LLOps.f_transpose_2d(b)
        return [[sumprod(a_i, b_T_j) for b_T_j in b_T] for a_i in a]

    @staticmethod
    def f_matmul_transposed_2d(a: list, bT: list) -> list:
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (J, K).T  -> (I,J)
        return [[sumprod(a_i, bT_j) for bT_j in bT] for a_i in a]

    @staticmethod
    def f_matmul_2d_multiprocess(a: list, b: list) -> list:
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        with Pool(8) as p:
            return p.starmap(LLOps.f_vec_times_mat, ((a_i, b) for a_i in a), chunksize=max(1, I//8))

    @staticmethod
    def f_matmul_2d_old(a: list, b: list) -> list:
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
    def f_vec_times_vec(a: list, b: list) -> (int, float):
        # perform vector times matrix multiplication on a 1-dimensional list and another 1-dimensional lists
        assert len(a) == len(b)
        return sumprod(a, b)

    @staticmethod
    def f_mat_times_vec(a: list, b: list) -> list:
        # perform matrix times vector multiplication on a 2-dimensional list and another 1-dimensional lists
        return [LLOps.f_vec_times_vec(row, b) for row in a]

    @staticmethod
    def f_vec_times_mat(a: list, b: list) -> list:
        # perform vector times vector multiplication on two 1-dimensional lists
        return [LLOps.f_vec_times_vec(a, row) for row in LLOps.f_transpose_2d(b)]

    @staticmethod
    def f_squeeze(a: list, dim: int) -> (list, int, float):
        # remove one dimension from a list of lists
        if dim == 0:
            return a[0]
        elif dim == 1:
            return [a_i[0] for a_i in a]
        else:
            return [LLOps.f_squeeze(a_i, dim-1) for a_i in a]

    @staticmethod
    def f_unsqueeze(a: list, dim: int) -> list:
        # add one dimension to a list of lists
        if dim == 0:
            return [a]
        elif dim == 1:
            return [[a_i] for a_i in a]
        else:
            return [LLOps.f_unsqueeze(a_i, dim-1) for a_i in a]

    @staticmethod
    def f_slice(a: list, item: tuple) -> list:
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
    def f_flatten(a: list, dim: int = 0) -> list:
        # Flatten a list of lists into a single list starting at dim
        if dim == 0:
            if isinstance(a[0], list):
                if isinstance(a[0][0], list):
                    return [subsublist for sublist in a for subsublist in LLOps.f_flatten(sublist)]
                else:
                    return [num for sublist in a for num in sublist]
            else:
                return a
        else:  # dim > 0
            return [LLOps.f_flatten(a_i, dim-1) for a_i in a]

    @staticmethod
    def f_calc_shape(a: list) -> list:
        assert isinstance(a, list)
        if not isinstance(a[0], list):
            return [len(a)]
        else:
            l = LLOps.f_calc_shape(a[0])
            l.insert(0, len(a))
            return l

    @staticmethod
    def f_setitem(a: list, key: tuple, value) -> list:
        # set the item at position key of list a to a value. Value can be scalar or list.  (a[key] = value)
        if len(key) == 1:
            a[key[0]] = value
            return a
        else:
            i = key[0]
            if isinstance(i, int):
                m = a[:i] + [LLOps.f_setitem(a[i], key[1:], value)] + a[i+1:]
            else:  # i is a slice
                assert i.step is None
                start = 0 if i.start is None else i.start if i.start > 0 else len(a) + i.start
                stop = len(a) if i.stop is None else i.stop if i.stop > 0 else len(a) + i.stop
                m = a[:start]
                for a_i_j, v_i in zip(a[i], value):
                    m += [LLOps.f_setitem(a_i_j, key[1:], v_i)]
                m += a[stop:]
            return m

    @staticmethod
    def f_reshape_flattened(a: list, shape: tuple) -> list:
        # reshape a one-dimensional array (flattened) into a target format
        if len(shape) == 1:
            return a
        else:
            n = len(a) // shape[0]  # 2
            return [LLOps.f_reshape_flattened(a[i*n:(i+1)*n], shape[1:]) for i in range(shape[0])]

    @staticmethod
    def f_permute_201(a: list) -> list:
        # permute shape (H, W, C) to (C, H, W)
        I, J, K = len(a), len(a[0]), len(a[0][0])

        m = [[[a[i][j][k]
               for j in range(J)]
              for i in range(I)]
             for k in range(K)]
        return m

    @staticmethod
    def f_permute_10(a: list) -> list:
        # permute shape (I, J, K) to (J, I, K)
        I, J = len(a), len(a[0])

        m = [[a[i][j]
               for i in range(I)]
              for j in range(J)]
        return m

    @staticmethod
    def f_advanced_indexing_1d(a: (list, tuple), b: (list, tuple)):
        return tuple([a[b_i] for b_i in b])

    @staticmethod
    def f_reduction_sum(a: list, reduction_dim: int, a_shape: tuple) -> (list, int, float):
        # sum up the list (of lists) along the dimensions specified in shape
        if reduction_dim == 0:
            if len(a_shape) == 1:
                return sum(a)
            else:
                zeros = LLOps.fill(a_shape[1:], 0)
                for a_i in a:  # zeros = ((zeros + a_0) + a_1) + ...
                    zeros = LLOps.f_binary_op_same_size(zeros, a_i, operator.add)
                return zeros
        else:
            return [LLOps.f_reduction_sum(a_i, reduction_dim - 1, a_shape[1:]) for a_i in a]

    @staticmethod
    def f_reduction_max(a: list, reduction_dim: int, a_shape: tuple) -> (list, int, float):
        # find the max of the list (of lists) along the dimensions specified in shape
        if reduction_dim == 0:
            if len(a_shape) == 1:
                return max(a)
            else:
                neg_inf = LLOps.fill(a_shape[1:], -math.inf)
                for a_i in a:
                    neg_inf = LLOps.f_binary_op_same_size(neg_inf, a_i, lambda x, y: max(x, y))
                return neg_inf
        else:
            return [LLOps.f_reduction_max(a_i, reduction_dim - 1, a_shape[1:]) for a_i in a]

    @staticmethod
    def f_reduction_prod(a: list, reduction_dim: int, a_shape: tuple) -> (list, int, float):
        # calculate the product of the list (of lists) along the dimensions specified in shape
        if reduction_dim == 0:
            if len(a_shape) == 1:
                return math.prod(a)
            else:
                inter = LLOps.fill(a_shape[1:], 1)
                for a_i in a:
                    inter = LLOps.f_binary_op_same_size(inter, a_i, operator.mul)
                return inter
        else:
            return [LLOps.f_reduction_prod(a_i, reduction_dim - 1, a_shape[1:]) for a_i in a]

    @staticmethod
    def f_stack(iterable: list, dim: int) -> list:
        # stack iterables of lists along a certain dim. NOT working yet
        if dim == 0:
            return [a_i for a_i in iterable]
        elif dim == 1:
            return [[elem for a_i in iterable
                    for elem in a_i[row_index]]
                    for row_index in range(len(iterable[0]))]
        else:
            raise NotImplementedError

    @staticmethod
    def f_cat(iterable: list, dim: int) -> list:
        # concatenate iterables of lists along a certain dim. ONLY working for dim 0 and 1
        if dim == 0:
            return [a_i_j for a_i in iterable for a_i_j in a_i]
        elif dim == 1:
            return [[a_i_j for a_i in iterable for a_i_j in a_i[row_index]]
                    for row_index in range(len(iterable[0]))]
        else:
            raise NotImplementedError


class Tensor:
    # Wrapper to use linalg operations on lists (of lists) (e.g. matmuls) in a nicer way

    def __init__(self, elems: (list, int, float)):
        if not isinstance(elems, (list, int, float)):  # e.g. numpy array or torch tensor
            elems = elems.tolist()
        self.elems = elems

        # calculate number of dimensions
        self.ndim = 0
        self.shape = []
        self._calc_shape_and_dims()

    def replace(self, new_tensor):
        self.__init__(new_tensor.elems)

    def _calc_shape_and_dims(self):
        shape_list = []
        a = self.elems
        while isinstance(a, list) and len(a) > 0:
            shape_list.append(len(a))
            a = a[0]
            self.ndim += 1
        self.shape = tuple(shape_list)

    def __repr__(self):
        elems_str = str(self.elems)
        if len(elems_str) > 200:
            elems_str = elems_str[:90] + "  ...  " + elems_str[-90:]
        return f"Tensor of shape {self.shape}.  Elements ({elems_str})"

    def __eq__(self, other):
        return self.elems == other.elems

    def size(self, dim=slice(None)):
        return self.shape[dim]

    ######################
    # Construction Methods
    #####################

    @staticmethod
    def zeros(*shape, dtype=float):
        if len(shape)==1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        value = 0.0 if dtype == float else 0 if dtype == int else None
        return Tensor.fill(shape, value=value)

    @staticmethod
    def ones(*shape, dtype=float):
        if len(shape)==1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        value = 1.0 if dtype == float else 1 if dtype == int else None
        return Tensor.fill(shape, value=value)

    @staticmethod
    def fill(*shape, value):
        if len(shape)==1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor(LLOps.fill(shape, value))

    @staticmethod
    def random_float(*shape, min=-1.0, max=1.0):
        if isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor(LLOps.fill_callable(shape, lambda: random.uniform(min, max)))

    @staticmethod
    def random_int(*shape, min=0, max=10):
        if isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor(LLOps.fill_callable(shape, lambda: random.randint(min, max)))

    @staticmethod
    def stack(tensor_list):
        return Tensor([t.elems for t in tensor_list])

    ########################
    # Arithmetic Operations
    #######################

    @staticmethod
    def _basic_op(a, b, op):
        # input tensor output list of list
        if isinstance(b, Tensor):
            # print(f"add/mul shapes {a.shape} and {b.shape}")
            if a.shape == ():
                return LLOps.f_binary_op_scalar(b.elems, a.item(), op)
            elif b.shape == ():
                return LLOps.f_binary_op_scalar(a.elems, b.item(), op)
            elif a.shape == b.shape:
                return LLOps.f_binary_op_same_size(a.elems, b.elems, op)
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
            return LLOps.f_binary_op_scalar(a.elems, b, op)
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

    def __truediv__(self, other):
        return Tensor(Tensor._basic_op(self, other, operator.truediv))

    def __floordiv__(self, other):
        return Tensor(Tensor._basic_op(self, other, operator.floordiv))

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

    def matmul_T_2d(self, other):
        # self.matmul_t_2d(other) is the same as self @ other.T
        return Tensor(LLOps.f_matmul_transposed_2d(self.elems, other.elems))

    @property
    def T(self):
        if self.ndim == 2:
            return Tensor(LLOps.f_transpose_2d(self.elems))
        else:
            raise NotImplementedError

    def abs(self):
        return Tensor(LLOps.f_unary_op(self.elems, abs))

    def __abs__(self):
        return self.abs()

    def sqrt(self):
        return Tensor(LLOps.f_unary_op(self.elems, math.sqrt))

    def exp(self):
        return Tensor(LLOps.f_unary_op(self.elems, math.exp))

    def pow(self, num):
        if num == 2:
            return Tensor(LLOps.f_binary_op_same_size(self.elems, self.elems, operator.mul))
        elif isinstance(num, int):
            return self.apply(lambda x: x**num)
        else:
            return self.apply(lambda x: math.pow(x, num))

    def __pow__(self, num):
        return self.pow(num)

    def clamp(self, low, high):
        if low is None:
            return Tensor(LLOps.f_unary_op(self.elems, lambda x: min(x, high)))
        elif high is None:
            return Tensor(LLOps.f_unary_op(self.elems, lambda x: max(x, low)))
        else:
            return Tensor(LLOps.f_unary_op(self.elems, lambda x: max(min(x, high), low)))

    def sum(self, dims=None):
        if isinstance(dims, int):
            return Tensor(LLOps.f_reduction_sum(self.elems, dims, self.shape))
        else:  # iterable (list, tuple) or None
            inter = self.elems
            dims_iter = range(self.ndim) if dims is None else dims
            for i, d in enumerate(dims_iter):
                inter = LLOps.f_reduction_sum(inter, d - i, self.shape[i:])
            return Tensor(inter)

    def mean(self, dims=None):
        count = 1
        if dims is None:
            count = math.prod(self.shape)
        elif isinstance(dims, int):
            count = self.shape[dims]
        else:  # iterable (list, tuple)
            for i in dims:
                count *= self.shape[i]
        return self.sum(dims) * (1 / count)

    def max(self, dims=None):
        if isinstance(dims, int):
            return Tensor(LLOps.f_reduction_max(self.elems, dims, self.shape))
        else:  # iterable (list, tuple) or None
            inter = self.elems
            dims_iter = range(self.ndim) if dims is None else dims
            for i, d in enumerate(dims_iter):
                inter = LLOps.f_reduction_max(inter, d - i, self.shape[i:])
            return Tensor(inter)

    def prod(self, dim):
        assert isinstance(dim, int) is True
        return Tensor(LLOps.f_reduction_prod(self.elems, dim, self.shape))

    def log(self):
        return Tensor(LLOps.f_unary_op(self.elems, math.log))

    def apply(self, func):
        return Tensor(LLOps.f_unary_op(self.elems, func))

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
        v = copy.deepcopy(v)
        if isinstance(key, (int, slice)):
            self.elems[key] = v
        else:  # key is tuple, list, or other iterable
            self.elems = LLOps.f_setitem(self.elems, key, v)

    def flatten(self, dim=0):
        return Tensor(LLOps.f_flatten(self.elems, dim))

    def tolist(self):
        return self.elems

    def item(self):
        return self.elems

    def reshape(self, shape):
        return Tensor(LLOps.f_reshape_flattened(LLOps.f_flatten(self.elems), shape))

    def view(self, shape):
        return self.reshape(shape)

    def permute(self, shape):
        # Important cases are handled much quicker
        if len(shape) >= 2 and shape[:2] == (1, 0) and shape[2:] == tuple(range(2, len(shape))):
            # e.g. permute(1,0,2,3,4,5,6,7)
            return Tensor(LLOps.f_permute_10(self.elems))
        if len(shape) >= 3 and shape[:3] == (2, 0, 1) and shape[3:] == tuple(range(3, len(shape))):
            # e.g. permute(2,0,1,3,4,5,6,7)
            return Tensor(LLOps.f_permute_201(self.elems))

        # newindex = old_index[perm]
        def calc_card(a):
            prod = 1
            card = []
            for n in reversed(a):
                card.insert(0, prod)
                prod *= n
            return card

        old_shape = self.shape
        new_shape = LLOps.f_advanced_indexing_1d(old_shape, shape)
        new_tensor = Tensor.zeros(new_shape)
        card = calc_card(old_shape)
        for i in range(math.prod(old_shape)):
            multi_dim_index = [(i//c) % m for c, m in zip(card, old_shape)]
            new_index = LLOps.f_advanced_indexing_1d(multi_dim_index, shape)
            new_tensor[new_index] = self[tuple(multi_dim_index)].item()
        return new_tensor


class nn:
    class Module:

        def __init__(self, *args, **kwargs):
            if args != () or kwargs != {}:
                print('Warning, ignoring', args, kwargs)
            self.cache = None

        def __call__(self, *x):
            return self.forward(*x)

        def eval(self):
            pass

        def train(self):
            pass

        def forward(self, x: Tensor):
            raise NotImplementedError

        def backward(self, dout: Tensor):
            raise NotImplementedError

        def load_state_dict(self, state_dict):
            weight_apply(self, state_dict)

    class ReLU(Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x: Tensor):
            self.cache = x
            return x.apply(lambda v: max(0.0, v))

        def backward(self, dout: Tensor):
            x = self.cache
            mask = x.apply(lambda v: 1 if v >= 0 else 0)
            dx = dout * mask
            return dx

    class LeakyReLU(Module):
        def __init__(self, negative_slope=0.01, *args, **kwargs):
            super().__init__()
            self.negative_slope = negative_slope

        def forward(self, x: Tensor):
            self.cache = x
            return x.apply(lambda v: self.negative_slope*v if v < 0 else v)

        def backward(self, dout: Tensor):
            x = self.cache
            mask = x.apply(lambda v: 1 if v >= 0 else self.negative_slope)
            dx = dout * mask
            return dx

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            init_value_min = -0.1
            init_value_max = 0.1
            self.weight = Tensor.random_float((out_features, in_features), min=init_value_min, max=init_value_max)
            if bias:
                self.bias = Tensor.random_float((out_features, ), min=init_value_min, max=init_value_max)
            else:
                self.bias = Tensor.zeros(out_features)
            self.dw = None
            self.db = None

        def forward(self, x: Tensor):
            # shapes x: (B, C_in) , w.T: (C_in, C_out)  b: (C_out)
            start = time.time()
            self.cache = x
            res = x.matmul_T_2d(self.weight) + self.bias
            print("Linear took ", time.time() - start)
            return res

        def backward(self, dout: Tensor):
            x = self.cache

            # grad wrt weights
            self.dw = x.T @ dout

            # grad wrt bias
            self.db = dout.sum(0)

            # grad wrt x
            dx = (dout @ self.weight).reshape(x.shape)
            return dx

        def update(self, lr):
            self.weight -= self.dw.T * lr
            self.bias -= self.db * lr

    class Sequential(Module):
        skip_cache = {}
        skip_grad_cache = {}

        def __init__(self, *modules):
            super().__init__()
            self.modules = modules

        def __getattr__(self, name):
            # does not work as intended
            if name.isnumeric():
                return self.modules[int(name)]

        def forward(self, x: Tensor):
            for module in self.modules:
                x = module(x)
            return x

        def backward(self, dout: Tensor):
            for module in reversed(self.modules):
                dout = module.backward(dout)
            return dout

        def update(self, lr):
            for module in self.modules:
                if hasattr(module, 'update'):
                    module.update(lr)

    class SkipStart(Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x: Tensor):
            nn.Sequential.skip_cache[self.name] = x
            return x

        def backward(self, dout: Tensor):
            return dout + nn.Sequential.skip_grad_cache[self.name]

    class SkipEnd(Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x: Tensor):
            return x + nn.Sequential.skip_cache.pop(self.name)

        def backward(self, dout: Tensor):
            nn.Sequential.skip_grad_cache[self.name] = dout
            return dout

    class Conv2d(Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=True, *args, **kwargs):
            if args != () or kwargs != {}:
                print('Warning, Conv2dTranspose ignoring', args, kwargs)
            super().__init__()
            assert dilation == 1, 'Dilation unequal to 1 not implemented'
            assert out_channels % groups == 0
            assert in_channels % groups == 0
            self.stride = stride
            self.padding = padding
            self.kernel_size = kernel_size
            self.out_channels = out_channels
            self.groups = groups
            self.weight = Tensor.fill((out_channels, in_channels//groups, kernel_size, kernel_size), value=0.0)
            self.bias = Tensor.fill((out_channels, ), value=0.0 if bias else 0.0)

        def forward(self, x: Tensor):
            if self.groups == 1:  # and dilation==1
                return self.forward_gemm(x)
            else:
                return self.forward_gemm_advanced(x)

        def forward_naive(self, x: Tensor):
            #  shapes x: (B, C_in, H, W)    w: (C_out, C_in, K, K)    b: (C_Out)    out: (B, C_out, ~H/s, ~W/s)
            B, C_in, H, W = x.shape
            C_out = self.out_channels
            K, P, S = self.kernel_size, self.padding, self.stride
            H_out = (H - K + 2 * P) // S + 1
            W_out = (W - K + 2 * P) // S + 1
            output_tensor = Tensor.zeros((B, C_out, H_out, W_out))
            x_padded = Tensor.zeros((B, C_in, H+2*P, W+2*P))
            x_padded[:, :, P:-P, P:-P] = x
            for h_out, h in enumerate(range(0, H + 2 * P - K + 1, S)):
                for w_out, w in enumerate(range(0, W + 2 * P - K + 1, S)):
                    for c_out in range(C_out):
                        x_chunk = x_padded[:, :, h:h+K, w:w+K]  # shape (B, C_in, K, K)
                        assert x_chunk.shape == (B, C_in, K, K), "wrong shape"
                        w_c = self.weight[c_out, :, :, :]  # shape (C_in, K, K)
                        output_tensor[:, c_out, h_out, w_out] = (x_chunk * w_c).sum((1, 2, 3)) + self.bias[c_out]
            assert output_tensor.shape == (B, C_out, H_out, W_out), "wrong output shape"
            return output_tensor

        def forward_gemm(self, x: Tensor):
            # Forward of Convolution layers with transforming to a single Matmul
            start_time = time.time()
            B, C_in, H, W = x.shape
            C_out = self.out_channels
            K, P, S = self.kernel_size, self.padding, self.stride
            H_out = (H - K + 2 * P) // S + 1
            W_out = (W - K + 2 * P) // S + 1

            x_padded = Tensor.zeros(B, C_in, H + 2 * P, W + 2 * P)
            x_padded[:, :, P:H+P, P:W+P] = x
            assert x_padded[:, :, P:H + P, P:W + P] == x

            reshaped_kernel = self.weight.reshape((C_out, C_in * K * K))

            def im2col(x_pad):
                coloums = [x_pad[b, :, h:h+K, w:w+K].flatten().tolist()
                           for b in range(B)
                           for h in range(0, H + 2 * P - K + 1, S)
                           for w in range(0, W + 2 * P - K + 1, S)
                           ]
                return coloums

            col_repres = im2col(x_padded)
            start_mat = time.time()
            res = reshaped_kernel.matmul_T_2d(Tensor(col_repres))  # for performance reasons, Equal to reshaped_kernel @ Tensor(col_repres).T
            end_mat = time.time()
            if B > 1:
                res = res.reshape((C_out, B, H_out, W_out)).permute((1,0,2,3))
            else:
                res = res.reshape((1, C_out, H_out, W_out))
            res = res + self.bias.reshape((1, C_out, 1, 1))

            assert res.shape == (B, C_out, H_out, W_out)
            print("Conv2d took in total", time.time() - start_time, " of which Matmul took", end_mat - start_mat)
            return res

        def forward_gemm_advanced(self, x: Tensor):
            # Forward of Convolution layers with transforming to a single Matmul. Supports grouped convs
            start_time = time.time()
            B, C_in, H, W = x.shape
            C_out = self.out_channels
            K, P, S, G = self.kernel_size, self.padding, self.stride, self.groups
            H_out = (H - K + 2 * P) // S + 1
            W_out = (W - K + 2 * P) // S + 1

            x_padded = Tensor.zeros((B, G, C_in // G, H + 2 * P, W + 2 * P))
            x_padded[:, :, :, P:H + P, P:W + P] = x.reshape((B, G, C_in // G, H, W))
            assert x_padded[:, :, :, P:H + P, P:W + P] == x.reshape((B, G, C_in // G, H, W))

            reshaped_kernel = self.weight.reshape((G, C_out // G, C_in // G * K * K))

            def im2col(x_pad):
                coloums = [[x_pad[b, g, :, h:h + K, w:w + K].flatten().tolist()
                           for b in range(B)
                           for h in range(0, H + 2 * P - K + 1, S)
                           for w in range(0, W + 2 * P - K + 1, S)
                           ] for g in range(G)]
                return coloums

            col_repres = im2col(x_padded)
            start_mat = time.time()
            res = Tensor.stack([reshaped_kernel[i].matmul_T_2d(Tensor(col_repres[i])) for i in range(G)])
            end_mat = time.time()
            res = res.reshape((G, C_out//G, B, H_out, W_out)).permute((2,0,1,3,4))
            res = res.reshape((B, C_out, H_out, W_out))
            res = res + self.bias.reshape((1, C_out, 1, 1))

            assert res.shape == (B, C_out, H_out, W_out), f"res shape {res.shape} is not {(B, C_out, H_out, W_out)}"
            print("grouped Conv2d took in total", time.time() - start_time, " of which Matmul took", end_mat - start_mat)
            return res

    class Conv2dTranspose(Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, bias=True, *args, **kwargs):
            if args != () or kwargs != {}:
                print('Warning, Conv2dTranspose ignoring', args, kwargs)
            super().__init__()
            self.stride = stride
            self.padding = padding
            self.kernel_size = kernel_size
            self.out_channels = out_channels
            self.weight = Tensor.fill((out_channels, in_channels, kernel_size, kernel_size), value=0.0)
            self.bias = Tensor.fill((out_channels, ), value=0.0 if bias else 0)

        def forward(self, x: Tensor):
            raise NotImplementedError

    class BatchNorm2d(Module):
        def __init__(self, num_features, eps=1e-05, *args, **kwargs):
            if args != () or kwargs != {}:
                print('Warning, BatchNorm2d ignoring', args, kwargs)
            self.weight = Tensor.fill((num_features,), value=0.0)
            self.bias = Tensor.fill((num_features,), value=0.0)
            self.running_mean = Tensor.fill((num_features,), value=0.0)
            self.running_var = Tensor.fill((num_features,), value=1.0)
            self.num_batches_tracked = Tensor([0.0])
            self.C = num_features
            self.eps = eps
            super().__init__()

        def forward(self, x: Tensor):
            start_time = time.time()
            C = self.C
            mean = self.running_mean.reshape((1, C, 1, 1))
            std = (self.running_var + self.eps).sqrt().reshape((1, C, 1, 1))
            weight = self.weight.reshape((1, C, 1, 1))
            bias = self.bias.reshape((1, C, 1, 1))
            y = ((x - mean) / std) * weight + bias

            assert y.shape == x.shape
            print("BatchNorm took ", time.time() - start_time)
            return y

    class MaxPool2d(Module):
        def __init__(self, kernel_size=2, stride=2, padding=0, *args, **kwargs):
            if args != () or kwargs != {}:
                print('Warning, MaxPool2d ignoring', args, kwargs)
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            super().__init__()

        def forward(self, x: Tensor):
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    class AdaptiveAvgPool2d(Module):
        def __init__(self, output_size):
            self.H_out = output_size[0]
            self.W_out = output_size[1]
            super().__init__()

        def forward(self, x: Tensor):
            start_time = time.time()
            B, C, H, W = x.shape
            stride_H = H // self.H_out
            stride_W = W // self.W_out
            kernel_H = H - (self.H_out - 1) * stride_H
            kernel_W = W - (self.W_out - 1) * stride_W

            out = Tensor([[[[x[b, c, new_h * stride_H:new_h * stride_H + kernel_H, new_w * stride_W:new_w * stride_W + kernel_W].mean().item()
                            for new_w in range(self.W_out)]
                           for new_h in range(self.H_out)]
                          for c in range(C)]
                         for b in range(B)])

            assert out.shape == (B, C, self.H_out, self.W_out)
            print("AdaptiveAvgPool2d took", time.time() - start_time)
            return out

    class Dropout(Module):
        def forward(self, x: Tensor):
            return x

    class Upsample(Module):
        def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False, *args, **kwargs):
            super().__init__()
            print('Warning, ignoring', args, kwargs)
            self.size = size
            self.scale_factor = scale_factor
            self.mode = mode
            self.align_corners = align_corners

        def forward(self, x: Tensor):
            return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

    class AbstractLoss:
        def __init__(self):
            self.cache = None

        def __call__(self, input: Tensor, target: Tensor):
            return self.forward(input, target)

    class MSELoss(AbstractLoss):
        def forward(self, input: Tensor, target: Tensor):
            inv_size = 1.0 / math.prod(input.shape)
            diff = input - target
            self.cache = inv_size, diff
            return (diff ** 2).sum() * inv_size

        def backward(self, dout: Tensor):
            inv_size, diff = self.cache
            dout = 2 * diff * inv_size * dout
            return dout

    class BCELoss(AbstractLoss):
        def forward(self, input: Tensor, target: Tensor):
            inv_size = 1.0 / math.prod(input.shape)
            self.cache = input, target, inv_size
            return -inv_size * (input * target.log() + (1-input) * (1-target).log())

        def backward(self, dout: Tensor):
            input, target, inv_size = self.cache
            return inv_size * (target - input) * dout


    # missing: weight_init, ConvTranspose2d, AvgPool2d, Softmax, InstanceNorm2d, LayerNorm2d, Losses


class F:
    @staticmethod
    def interpolate(input: Tensor, size=None, scale_factor=None, mode='nearest', align_corners=False):
        # TODO: make bilinear work with align_corners=False
        B, C, H, W = input.shape
        time_start = time.time()
        assert mode in ['nearest', 'bilinear']

        def get_new_size_and_scale_factors(size, scale_factor):
            if size is None:
                if isinstance(scale_factor, tuple):
                    scale_factor_h, scale_factor_w = scale_factor[-2:]
                else:  # scale_factor is float
                    scale_factor_h = scale_factor
                    scale_factor_w = scale_factor
                new_H = int(math.floor(H * scale_factor_h))
                new_W = int(math.floor(W * scale_factor_w))
                if align_corners:
                    scale_factor_h = (new_H - 1) / (H - 1)
                    scale_factor_w = (new_W - 1) / (W - 1)
            else:
                if isinstance(size, int):
                    size = (size, size)
                new_H, new_W = size
                su = 1 if align_corners else 0
                scale_factor_h = (new_H - su) / (H - su)
                scale_factor_w = (new_W - su) / (W - su)
            return new_H, new_W, scale_factor_h, scale_factor_w

        new_H, new_W, scale_factor_h, scale_factor_w = get_new_size_and_scale_factors(size, scale_factor)
        output_tensor = Tensor.fill((B, C, new_H, new_W), value=0.0)
        if mode == 'nearest':
            for new_h in range(new_H):
                for new_w in range(new_W):
                    old_h = int(new_h // scale_factor_h)
                    old_w = int(new_w // scale_factor_w)
                    output_tensor[:, :, new_h, new_w] = input[:, :, old_h, old_w]
        elif mode == 'bilinear':
            for new_h in range(new_H):
                for new_w in range(new_W):
                    old_h_low = int(new_h // scale_factor_h)
                    old_w_low = int(new_w // scale_factor_w)
                    old_h_high = max(0, min(old_h_low + 1, H-1))
                    old_w_high = max(0, min(old_w_low + 1, W-1))
                    frac_h = new_h / scale_factor_h - old_h_low
                    frac_w = new_w / scale_factor_w - old_w_low
                    p_top_left = (1 - frac_h) * (1 - frac_w) * (input[:, :, old_h_low, old_w_low])
                    p_top_right = (1 - frac_h) * frac_w * (input[:, :, old_h_low, old_w_high])
                    p_bottom_left = frac_h * (1 - frac_w) * (input[:, :, old_h_high, old_w_low])
                    p_bottom_right = frac_h * frac_w * (input[:, :, old_h_high, old_w_high])
                    output_tensor[:, :, new_h, new_w] = p_top_left + p_top_right + p_bottom_left + p_bottom_right
        print("interpolate took ", time.time() - time_start)
        return output_tensor

    @staticmethod
    def relu(x: Tensor, inplace = False):
        return x.apply(lambda v: max(0.0, v))

    @staticmethod
    def max_pool2d(x: Tensor, kernel_size, stride, padding=0, *args, **kwargs):
        B, C_in, H, W = x.shape
        start_time = time.time()
        K, P, S = kernel_size, padding, stride
        H_out = (H - K + 2 * P) // S + 1
        W_out = (W - K + 2 * P) // S + 1
        x_padded = Tensor.fill((B, C_in, H + 2 * P, W + 2 * P), value=-math.inf)
        x_padded[:, :, P:H + P, P:W + P] = x
        assert x_padded[:, :, P:H + P, P:W + P].tolist() == x.tolist()
        output_tensor = Tensor.zeros((B, C_in, H_out, W_out))
        for h_out, h in enumerate(range(0, H + 2 * P - K + 1, S)):
            for w_out, w in enumerate(range(0, W + 2 * P - K + 1, S)):
                output_tensor[:, :, h_out, w_out] = x_padded[:, :, h:h + K, w:w + K].max((2, 3))
        print("MaxPool took ", time.time() - start_time)
        return output_tensor

    @staticmethod
    def softmax(input: Tensor, dim=0):
        if dim == 0:
            assert input.ndim == 1
            input_exp = (input - input.max()).exp()
            return input_exp / input_exp.sum()
        else:
            raise NotImplementedError

class PthUnpickler(pickle.Unpickler):
    def __init__(self, picklefile, zipfile, name):
        self.zipfile = zipfile
        self.name = name
        super().__init__(picklefile)

    def persistent_load(self, pid):
        # print("pid", pid)
        storage, class_type, storage_dir, gpu, size = pid
        data_list = None
        with self.zipfile.open(f'{self.name}/data/{storage_dir}') as f:
            data = f.read()
            if class_type in ["f", "q", "i"]:
                data_list = list(struct.unpack(f'<{size}{class_type}', data))
        return data_list

    @staticmethod
    def load_replacement(data_list, storage_offset, size, stride, *args):
        assert storage_offset == 0
        # print("storage:", "--", "size:", size, "stride:", stride)
        tensor = Tensor(data_list)
        if len(size) > 1:
            tensor = tensor.reshape(size)
        return tensor

    def find_class(self, module, name, strict=True):
        # print("m", module, "n", name)
        if module == 'torch' and name == 'FloatStorage':
            return "f"
        elif module == 'torch' and name == 'LongStorage':
            return "q"
        elif module == 'torch' and name == 'IntStorage':
            return "i"
        elif module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return self.load_replacement  # torch._utils._rebuild_tensor_v2
        elif module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        else:
            print("WARNING: loading module", module, "name", name)
            if strict:
                return super().find_class(module, name)
            else:
                return None


def load(path):
    with ZipFile(path) as zip_file:
        name = path.split("//")[-1][:-4]
        with zip_file.open(f'{name}/data.pkl') as pickle_file:
            print("loading ...")
            p = PthUnpickler(pickle_file, zip_file, name)
            model_dict = p.load()
            print("loading completed")
    return model_dict


def weight_apply(model, dict):
    for k, v in dict.items():
        current_a = model
        for part in k.split("."):
            if part.isnumeric():
                current_a = current_a.modules[int(part)]
            else:
                current_a = current_a.__getattribute__(part)
        if v.shape != current_a.shape:
            print(f"WARNING, change shape of {k} from {current_a.shape} to {v.shape}")
        current_a.replace(v)


class utils:
    @staticmethod
    def cat(tensors, dim):
        return Tensor(LLOps.f_cat([tensor.elems for tensor in tensors], dim))


    @staticmethod
    def topk(input: Tensor, k):
        assert input.ndim == 1
        el = input.elems
        el_sorted_with_index = sorted(zip(el, range(len(el))), key=lambda x: x[0], reverse=True)
        topk_prob = [v for (v, i) in el_sorted_with_index[:k]]
        topk_catid = [i for (v, i) in el_sorted_with_index[:k]]
        return topk_prob, topk_catid


class ImageIO:
    @staticmethod
    def png_filter(method, a, b, c):
        if method == 1:
            return a
        elif method == 3:
            return (a + b) // 2
        elif method == 4:
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            Pr = a if pa <= pb and pa <= pc else b if pb <= pc else c
            return Pr

    @staticmethod
    def png_decompress(data_bytes, width, height, n_channels):

        data_bytes = zlib.decompress(data_bytes)

        lines = []
        last_line = [[0 for _ in range(n_channels)] for _ in range(width + 1)]
        for line_idx in range(height):

            line_start = line_idx * (width * n_channels + 1)
            line_end = (line_idx+1) * (width * n_channels + 1)

            #line_flat_old = [int.from_bytes(data_bytes[i:i+1], "big") for i in range(line_start, line_end)]
            line_flat = list(struct.unpack(f'>{line_end - line_start}B', data_bytes[line_start:line_end]))

            #assert line_flat_old == line_flat, f"old \n{line_flat_old[-10:]}, new \n{line_flat[-10:]}"

            method = line_flat[0]
            line_flat = [0 for _ in range(n_channels)] + line_flat[1:]  # add zero padding for filtering
            line = LLOps.f_reshape_flattened(line_flat, (width + 1, n_channels))

            if method == 2:
                line = LLOps.f_binary_op_same_size(line, last_line, operator.add)  # line + last_line
                line = [[a_i_j % 256 for a_i_j in a_i] for a_i in line]           # line = line%256

            elif method == 1 or method == 3 or method == 4:
                for w in range(1, width + 1):
                    line[w] = [(line[w][c] + ImageIO.png_filter(method,
                                                                a=line[w-1][c],
                                                                b=last_line[w][c],
                                                                c=last_line[w-1][c])) % 256
                               for c in range(n_channels)]
            last_line = line
            lines.append(line[1:])
        return lines

    @staticmethod
    def read_png(path, resize=None, dimorder="HWC", num_channels=3, to_float=False, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        with open(path, "br") as f:
            image_bytes = f.read()
        data = b''
        width, height, bit_depth, color_type_str, color_type_bytes = None, None, None, None, None

        # header = image_bytes[:8]
        start = 8
        while start is not None:
            chunk_length = int.from_bytes(image_bytes[start:start+4], byteorder="big")
            chunk_type = image_bytes[start+4:start+8]
            chunk_data = image_bytes[start+8:start+8+chunk_length]
            if chunk_type == b'IHDR':
                width, height = int.from_bytes(chunk_data[:4], "big"), int.from_bytes(chunk_data[4:8], "big")
                bit_depth, color_type = int.from_bytes(chunk_data[8:9], "big"), int.from_bytes(chunk_data[9:10], "big")
                color_type_str = "RGB" if color_type == 2 else "GRAY" if color_type == 0 else "RGBA" if color_type == 6 else "OTHER"
                color_type_bytes = 3 if color_type == 2 else 1 if color_type == 0 else 4 if color_type == 6 else None
                assert bit_depth == 8

            if chunk_type == b'IDAT':
                data += chunk_data

            start = start + 12 + chunk_length

            if chunk_type == b'IEND':
                start = None
        lines = ImageIO.png_decompress(data, width, height, color_type_bytes)
        t = Tensor(lines)

        if t.shape[2] > num_channels:
            t = t[:, :, :num_channels]
        elif t.shape[2] < num_channels:
            raise NotImplementedError

        if resize:
            t = ImageIO.resize(t, resize)

        if to_float:
            t = t.apply(lambda x: (float(x) / 255.0))
            t = (t - Tensor([mean])) / Tensor([std])

        if dimorder == "HWC":
            pass
        elif dimorder == "BHWC":
            t = t.unsqueeze(0)
        elif dimorder == "CHW":
            t = t.permute((2,0,1))
        elif dimorder == "BCHW":
            t = t.permute((2,0,1)).unsqueeze(0)
        return t

    @staticmethod
    def save_png(path, tensor):
        raise NotImplementedError
        height, width, channels = tensor.shape
        data_compressed = zlib.compress(tensor.tolist())  # filter before that (line[w][c] - ImageIO.png_defilter) % 256
        header = b'\x89PNG\r\n\x1a\n'

        # IHDR
        h_chunk_size = int.to_bytes(13, length=4, byteorder="big")
        h_chunk_type = b'IHDR'
        h_chunk_data = ...
        h_chunk_crc = ...

        # IDATA
        d_chunk_size = ...
        d_chunk_type = b'IDATA'
        d_chunk_data = data_compressed
        d_chunk_crc = ...

        # IEND
        e_chunk_size = int.to_bytes(0, length=4, byteorder="big")
        e_chunk_type = b'IEND'
        e_chunk_data = b''
        e_chunk_crc = ...
        joint = (header + h_chunk_size + h_chunk_type + h_chunk_data + h_chunk_crc
                        + d_chunk_size + d_chunk_type + d_chunk_data + d_chunk_crc
                        + e_chunk_size + e_chunk_type + e_chunk_data + e_chunk_crc)

    @staticmethod
    def resize(tensor, new_size: tuple):
        H, W, C = tensor.shape
        new_tensor = tensor.zeros(new_size, dtype=int)
        for new_i, i in zip(range(new_size[0]), range(0, H, H // new_size[0])):
            for new_j, j in zip(range(new_size[1]), range(0, W, W // new_size[1])):
                new_tensor[new_i, new_j] = tensor[i, j]
                # TODO Fix
        return new_tensor


if __name__ == "__main__":
    try:
        a_read = ImageIO.read_png("source.png")
        print(Tensor(a_read))
    except FileNotFoundError:
        print("FILE NOT FOUND; continue")

    a_list = [[2, 3],
              [32, -21],
              [32, 21]]  # 3,2

    a_reshaped = Tensor(a_list).reshape((2, 3))
    print("SUM", a_reshaped.sum())
    print("SUM DIM 1", a_reshaped.sum(1))
    print("SHAPE", a_reshaped.shape)
    print("ABS", a_reshaped.abs())

    print(a_reshaped.permute((1, 0)))

    rand_tensor = Tensor.random_float((2, 4, 8, 2), 0.0, 1.2)
    print(rand_tensor)

    b_list = [[4, 2],
              [12, 22]]          # 2,2
    print("matmul of lists", LLOps.f_matmul_2d(a_list, b_list))
    print("matmul of lists (multiproc)", LLOps.f_matmul_2d_multiprocess(a_list, b_list))

    a_tensor = Tensor.zeros((2, 4, 8, 2))
    b_tensor = Tensor.zeros((2, 8))
    c_tensor = a_tensor@b_tensor  # shape (2, 4, 8, 8)
    d_tensor = c_tensor[0, 2:, :, :1] + b_tensor.unsqueeze(2)  # shape (2,8,1)
    print(d_tensor)

    # nn
    model = nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.SkipStart("a"),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.SkipEnd("a"),
        nn.Linear(128, 10),
        nn.LeakyReLU(),
    )
    input_tensor = Tensor.random_float((8, 20))
    output_tensor = model(input_tensor)
    target_tensor = Tensor.ones(output_tensor.shape)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output_tensor, target_tensor)
    print("output", output_tensor, "loss", loss)

    dout = loss_fn.backward(loss)
    dout = model.backward(dout)
    model.update(lr=0.01)

    print("grad dx/dd", dout)

__author__ = "Manuel Vogel"
__version__ = "0.0.1"
__website__ = "https://github.com/manu12121999/ctrl_c_nn"
__original_source__ = "https://github.com/manu12121999/ctrl_c_nn/blob/main/ctrl_c_nn.py"
__email__ = "manu12121999@gmail.com"

import collections
import pickle
import random
import struct
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
    def fill(shape: tuple, value):
        # Make a new list (of lists) filled with value
        if len(shape) == 1:
            return [value for _ in range(shape[0])]
        else:
            return [LLOps.fill(shape[1:], value) for _ in range(shape[0])]

    @staticmethod
    def fill_callable(shape: tuple, gen):
        # Make a new list (of lists) filled with values generated from the callable gen
        if len(shape) == 1:
            return [gen() for _ in range(shape[0])]
        else:
            return [LLOps.fill(shape[1:], gen()) for _ in range(shape[0])]

    @staticmethod
    def f_unary_op(a: list, f):
        # input tensor output list of list
        if isinstance(a[0], list):
            return [LLOps.f_unary_op(a_i, f) for a_i in a]
        else:
            return [f(a_i) for a_i in a]

    @staticmethod
    def f_operator_scalar(a: list, b: (int, float), op):
        # Add a scalar to a list (of lists). Other operations than add are supported too
        if isinstance(a, (int, float)):
            return op(a, b)
        elif isinstance(a[0], (int, float)):
            return [op(a_i, b) for a_i in a]
        else:
            return [LLOps.f_operator_scalar(a_i, b, op) for a_i in a]

    @staticmethod
    def f_operator_same_size(a: list, b: list, op):
        # Add two list (of lists). Other operations than add are supported too
        if isinstance(a[0], (int, float)):
            return [op(a_i, b_i) for a_i, b_i in zip(a, b)]
        else:
            return [LLOps.f_operator_same_size(a_i, b_i, op) for a_i, b_i in zip(a, b)]

    @staticmethod
    def f_add_same_size_performance(a: list, b: list):
        # Add two list (of lists). Only used to test the performance of different implementations
        if isinstance(a[0], (int, float)):
            return [a_i + b_i for a_i, b_i in zip(a, b)]
        else:
            return [LLOps.f_add_same_size_performance(a_i, b_i) for a_i, b_i in zip(a, b)]

    @staticmethod
    def f_transpose_2d(a: list):
        # Transpose a 2-dimensional list
        # (I,J) -> (J,I)
        I, J = len(a), len(a[0])
        return [[a[i][j] for i in range(I)] for j in range(J)]

    @staticmethod
    def f_clone_2d(a: list):
        # Deep-copy a 2-dimensional list of shape (I, J)
        I, J = len(a), len(a[0])
        return [[a[i][j] for j in range(J)] for i in range(I)]

    @staticmethod
    def f_matmul_2d(a: list, b: list):
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        b_T = LLOps.f_transpose_2d(b)
        return [[sumprod(a[i], b_T[j]) for j in range(J)] for i in range(I)]

    @staticmethod
    def f_matmul_2d_multiprocess(a: list, b: list):
        # perform matrix multiplication on two 2-dimensional lists
        # (I,K) @ (K, J)  -> (I,J)
        I, K, K2, J = len(a), len(a[0]), len(b), len(b[0])
        assert K == K2
        with Pool(24) as p:
            return p.starmap(LLOps.f_vec_times_mat, ((a_i, b) for a_i in a), chunksize=max(1, I//24))

    @staticmethod
    def f_matmul_2d_old(a: list, b: list):
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
    def f_vec_times_vec(a: list, b: list):
        # perform vector times matrix multiplication on a 1-dimensional list and another 2-dimensional lists
        assert len(a) == len(b)
        return sumprod(a, b)

    @staticmethod
    def f_mat_times_vec(a: list, b: list):
        # perform matrix times vector multiplication on a 2-dimensional list and another 1-dimensional lists
        return [LLOps.f_vec_times_vec(row, b) for row in a]

    @staticmethod
    def f_vec_times_mat(a: list, b: list):
        # perform vector times vector multiplication on two 1-dimensional lists
        return [LLOps.f_vec_times_vec(a, row) for row in LLOps.f_transpose_2d(b)]

    @staticmethod
    def f_squeeze(a: list, dim: int):
        # remove one dimension from a list of lists
        if dim == 0:
            return a[0]
        elif dim == 1:
            return [a_i[0] for a_i in a]
        else:
            return [LLOps.f_squeeze(a_i, dim-1) for a_i in a]

    @staticmethod
    def f_unsqueeze(a: list, dim: int):
        # remove one dimension from a list of lists
        if dim == 0:
            return [a]
        elif dim == 1:
            return [[a_i] for a_i in a]
        else:
            return [LLOps.f_unsqueeze(a_i, dim-1) for a_i in a]

    @staticmethod
    def f_slice(a: list, item: tuple):
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
    def f_flatten(a: list):
        # Flatten a list of lists into a single list
        if isinstance(a[0], list):
            if isinstance(a[0][0], list):
                return [subsublist for sublist in a for subsublist in LLOps.f_flatten(sublist)]
            else:
                return [num for sublist in a for num in sublist]
        else:
            return a

    @staticmethod
    def f_setitem(a: list, key: tuple, value):
        # set the item at position key of list a to a value. Value can be scalar or list.  (a[key] = value)
        if len(key) == 1:
            a[key[0]] = value
        else:
            LLOps.f_setitem(a[key[0]], key[1:], value)

    @staticmethod
    def f_reshape_flattened(a: list, shape: tuple):
        # reshape a one-dimensional array (flattened) into a target format
        if len(shape) == 1:
            return a
        else:
            n = len(a) // shape[0]  # 2
            return [LLOps.f_reshape_flattened(a[i*n:(i+1)*n], shape[1:]) for i in range(shape[0])]

    @staticmethod
    def f_advanced_indexing_1d(a: (list, tuple), b: (list, tuple)):
        return tuple([a[b_i] for b_i in b])

    @staticmethod
    def f_reduction_sum(a, dim, shape):
        if dim == 0:
            if len(shape) == 1:
                return sum(a)
            else:
                zeros = LLOps.fill(shape[1:], 0)
                for a_i in a:
                    zeros = LLOps.f_operator_same_size(zeros, a_i, operator.add)
                return zeros
        else:
            return [LLOps.f_reduction_sum(a_i, dim-1, shape[1:]) for a_i in a]

    @staticmethod
    def f_reduction_prod(a, dim, shape):
        if dim == 0:
            if len(shape) == 1:
                return math.prod(a)
            else:
                inter = LLOps.fill(shape[1:], 1)
                for a_i in a:
                    inter = LLOps.f_operator_same_size(inter, a_i, operator.mul)
                return inter
        else:
            return [LLOps.f_reduction_prod(a_i, dim-1, shape[1:]) for a_i in a]


class Tensor:
    # Wrapper to use linalg operations on lists (of lists) (e.g. matmuls) in a nicer way

    def __init__(self, elems):
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

    @staticmethod
    def random_float(shape, min=-1.0, max=1.0):
        if isinstance(shape, int):
            shape = (shape, )
        return Tensor(LLOps.fill_callable(shape, lambda: random.uniform(min, max)))

    @staticmethod
    def random_int(shape, min=0, max=10):
        if isinstance(shape, int):
            shape = (shape,)
        return Tensor(LLOps.fill_callable(shape, lambda: random.randint(min, max)))

    @staticmethod
    def stack(tensor_list):

        Tensor([t.elems for t in tensor_list])

    ########################
    # Arithmetic Operations
    #######################

    @staticmethod
    def _basic_op(a, b, op):
        # input tensor output list of list
        if isinstance(b, Tensor):
            # print(f"add/mul shapes {a.shape} and {b.shape}")
            if a.shape == ():
                return LLOps.f_operator_scalar(b.elems, a.item(), op)
            elif b.shape == ():
                return LLOps.f_operator_scalar(a.elems, b.item(), op)
            elif a.shape == b.shape:
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

    def __pow__(self, num):
        if num == 2:
            return Tensor(LLOps.f_operator_same_size(self.elems, self.elems, operator.mul))
        elif isinstance(num, int):
            return self.apply(lambda x: x**num)
        else:
            return self.apply(lambda x: math.pow(x, num))

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

    def view(self, shape):
        return self.reshape(shape)

    def permute(self, shape):
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
            self.cache = None

        def __call__(self, x: Tensor):
            return self.forward(x)

        def forward(self, x: Tensor):
            raise NotImplementedError

        def backward(self, dout: Tensor):
            raise NotImplementedError

    class ReLU(Module):
        def forward(self, x: Tensor):
            self.cache = x
            return x.apply(lambda v: max(0, v))

        def backward(self, dout: Tensor):
            x = self.cache
            mask = x.apply(lambda v: 1 if v >= 0 else 0)
            dx = dout * mask
            return dx

    class LeakyReLU(Module):
        def forward(self, x: Tensor):
            self.cache = x
            return x.apply(lambda v: 0.1*v if v < 0 else v)

        def backward(self, dout: Tensor):
            x = self.cache
            mask = x.apply(lambda v: 1 if v >= 0 else 0.1)
            dx = dout * mask
            return dx

    class Linear(Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            init_value_min = -0.1
            init_value_max = 0.1
            self.weight = Tensor.random_float(shape=(out_features, in_features), min=init_value_min, max=init_value_max)
            self.bias = Tensor.random_float(shape=(out_features, ), min=init_value_min, max=init_value_max)

            self.dw = None
            self.db = None

        def forward(self, x: Tensor):
            # shapes x: (B, C_in) , w.T: (C_in, C_out)  b: (C_out)
            self.cache = x
            return x @ self.weight.T + self.bias

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
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, bias=True):
            super().__init__()
            init_value_min = - 0.1
            init_value_max = 0.1
            self.stride = stride
            self.padding = padding
            self.kernel_size = kernel_size
            self.out_channels = out_channels
            self.weight = Tensor.random_float(shape=(out_channels, in_channels, kernel_size, kernel_size), min=init_value_min, max=init_value_max)
            self.bias = Tensor.random_float(shape=(out_channels, ), min=init_value_min, max=init_value_max) if bias else [0 for _ in range(out_channels)]

        def __call__(self, x: Tensor):
            return self.forward(x)

        def forward(self, x: Tensor):
            return self.forward_gemm(x)

        def forward_naive(self, x: Tensor):
            #  shapes x: (B, C_in, H, W)    w: (C_out, C_in, K, K)    b: (C_Out)    out: (B, C_out, H/s, W/s)
            B, C_in, H, W = x.shape
            C_out = self.out_channels
            K, P, S = self.kernel_size, self.padding, self.stride
            H_out = (H - K + 2 * P) // S + 1
            W_out = (W - K + 2 * P) // S + 1
            output_tensor = Tensor.zeros((B, C_out, H_out, W_out))
            x_padded = Tensor.zeros((B, C_in, H+2*P, W+2*P))
            x_padded[:, :, P:-P, P:-P] = x
            for h_out, h in enumerate(range(0, H + 2 * P - K - 1, S)):  # todo replace with + ?
                for w_out, w in enumerate(range(0, W + 2 * P - K - 1, S)):
                    for c_out in range(C_out):
                        x_chunk = x[:, :, h:h+K, w:w+K]  # shape (B, C_in, K, K)
                        w_c = self.weight[c_out, :, :, :]  # shape (C_in, K, K)
                        output_tensor[0, c_out, h_out, w_out] = (x_chunk * w_c).sum((1, 2, 3)) + self.bias[c_out]  #TODO: replace 0 by :
            return output_tensor

        def forward_gemm(self, x: Tensor):
            # Not working yet
            B, C_in, H, W = x.shape
            images = []
            for b in range(B):
                C_out = self.out_channels
                K, P, S = self.kernel_size, self.padding, self.stride
                H_out = (H - K + 2 * P) // S + 1
                W_out = (W - K + 2 * P) // S + 1
                reshaped_kernel = self.weight.reshape((C_out, C_in * K * K))

                def im2col(x, reshaped_size):
                    coloums = []
                    coloum_idx = 0
                    for h_out, h in enumerate(range(0, H + 2 * P - K + 1, S)):
                        for w_out, w in enumerate(range(0, W + 2 * P - K + 1, S)):
                            coloums.append(x[b, :, h:h+K, w:w+K].flatten().tolist())
                            coloum_idx += 1
                    return coloums

                col_repres = im2col(x, reshaped_size=(H_out*W_out, K*K*C_out))
                res = Tensor(col_repres) @ reshaped_kernel.T
                res = res.reshape((C_out, H_out, W_out))

                assert res.shape == (C_out, H_out, W_out)
                images.append(res)
            return Tensor.stack(images)

    class BatchNorm2d(Module):
        def __init__(self, num_features, *args, **kwargs):
            self.weight = Tensor.random_float((num_features,))
            self.bias = Tensor.random_float((num_features,))
            self.running_mean = Tensor.zeros((num_features,))
            self.running_var = Tensor.ones((num_features,))
            self.num_batches_tracked = Tensor([0])
            super().__init__(*args, **kwargs)

        def forward(self, x: Tensor):
            print("Warning: NOT IMPLEMENTED")
            return x

    class MaxPool2d(Module):
        def __init__(self, kernel_size=2, stride=2, padding=0):
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            super().__init__()

        def forward(self, x: Tensor):
            B, C_in, H, W = x.shape
            K, P, S = self.kernel_size, self.padding, self.stride
            H_out = (H - K + 2 * P) // S + 1
            W_out = (W - K + 2 * P) // S + 1
            output_tensor = Tensor.zeros((B, C_in, H_out, W_out))
            for h_out, h in enumerate(range(-P, H + P, S)):
                for w_out, w in enumerate(range(-P, W + P, S)):
                    output_tensor[:, :, h_out, w_out] = x[:, :, h:h+K, w:w+K].mean((2, 3))
            return x

    class AdaptiveAvgPool2d(Module):
        def forward(self, x: Tensor):
            print("Warning: NOT IMPLEMENTED")
            return x

    class AbstactLoss:
        def __init__(self):
            self.cache = None

        def __call__(self, input: Tensor, target: Tensor):
            return self.forward(input, target)

    class MSELoss(AbstactLoss):
        def forward(self, input: Tensor, target: Tensor):
            inv_size = 1.0 / math.prod(input.shape)
            diff = input - target
            self.cache = inv_size, diff
            return (diff ** 2).sum() * inv_size

        def backward(self, dout: Tensor):
            inv_size, diff = self.cache
            dout = 2 * diff * inv_size * dout
            return dout

    class BCELoss(AbstactLoss):
        def forward(self, input: Tensor, target: Tensor):
            inv_size = 1.0 / math.prod(input.shape)
            self.cache = input, target, inv_size
            return -inv_size * (input * target.log() + (1-input) * (1-target).log())

        def backward(self, dout: Tensor):
            input, target, inv_size = self.cache
            return inv_size * (target - input) * dout


    # missing: weight_init, ConvTranspose2d, AvgPool2d, Softmax, InstanceNorm2d, LayerNorm2d, Losses

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
            if class_type == "int64":
                data_list = [int.from_bytes(data[i: i+8], "little") for i in range(0, len(data), 8)]
            elif class_type == "int32":
                data_list = [int.from_bytes(data[i: i+4], "little") for i in range(0, len(data), 4)]
            elif class_type == "float":
                data_list = [struct.unpack('f', data[i: i+4])[0] for i in range(0, len(data), 4)]
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
            return "float"
        elif module == 'torch' and name == 'LongStorage':
            return "int64"
        elif module == 'torch' and name == 'IntStorage':
            return "int32"
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


def load_model(path):
    with ZipFile(path) as zip_file:
        name = path.split("//")[-1][:-4]
        with zip_file.open(f'{name}/data.pkl') as pickle_file:
            p = PthUnpickler(pickle_file, zip_file, name)
            print("loading ...")
            model_dict = p.load()
            print("loading completed")
            #print(model_dict)
    if "model_state_dict" in model_dict:
        return model_dict
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
        last_line = [[0 for _ in range(n_channels)] for _ in range(width)]
        for line_idx in range(height):
            line_start = line_idx * (width * n_channels + 1)
            line_end = (line_idx+1) * (width * n_channels + 1)

            method = int.from_bytes(data_bytes[line_start:line_start + 1], "big")
            line_flat = [int.from_bytes(data_bytes[i:i+1], "big") for i in range(line_start+1, line_end)]

            line = LLOps.f_reshape_flattened(line_flat, (width, n_channels))

            if method == 2:
                line = LLOps.f_operator_same_size(line, last_line, operator.add)  # line + last_line
                line = [[a_i_j % 256 for a_i_j in a_i] for a_i in line]           # line = line%256
            elif method == 1 or method == 3 or method == 4:
                for w in range(width):
                    line[w] = [(line[w][c] + ImageIO.png_filter(method,
                                                                a=line[w-1][c] if w > 0 else 0,
                                                                b=last_line[w][c],
                                                                c=last_line[w-1][c] if w > 0 else 0)) % 256
                               for c in range(n_channels)]
            last_line = line
            lines.append(line)
        return lines

    @staticmethod
    def read_png(path):
        with open(path, "br") as f:
            data = b''
            width, height, bit_depth, color_type_str, color_type_bytes = None, None, None, None, None

            image_bytes = f.read()
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
        return Tensor(lines)

    @staticmethod
    def save_png(path, tensor):
        raise NotImplementedError
        height, width, channels = tensor.shape
        data_compressed = zlib.compress(tensor.tolist())  # filter before that (line[w][c] - ImageIO.png_defilter) % 256
        header = b'\x89PNG\r\n\x1a\n'

        # IHDR
        h_chunk_size = int.to_bytes(13)
        h_chunk_type = b'IHDR'
        h_chunk_data = ...
        h_chunk_crc = ...

        # IDATA
        d_chunk_size = ...
        d_chunk_type = b'IDATA'
        d_chunk_data = data_compressed
        d_chunk_crc = ...

        # IEND
        e_chunk_size = int.to_bytes(0)
        e_chunk_type = b'IEND'
        e_chunk_data = b''
        e_chunk_crc = ...
        joint = (header + h_chunk_size + h_chunk_type + h_chunk_data + h_chunk_crc
                        + d_chunk_size + d_chunk_type + d_chunk_data + d_chunk_crc
                        + e_chunk_size + e_chunk_type + e_chunk_data + e_chunk_crc)


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

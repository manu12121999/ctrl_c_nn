import random
import sys
import math
import operator
from multiprocessing import Pool
import itertools

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
        if isinstance(a[0], (int, float)):
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
                return LLOps.f_operator_scalar(a.elems, b[0], op)
            elif b.shape == ():
                return LLOps.f_operator_scalar(a.elems[0], b, op)
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

    def sum(self, dim=None):
        if dim is None:
            inter = self.elems
            for i in range(self.ndim):
                inter = LLOps.f_reduction_sum(inter, 0, self.shape[i:])
            return Tensor(inter)
        else:
            assert isinstance(dim, int) is True
            return Tensor(LLOps.f_reduction_sum(self.elems, dim, self.shape))

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

        def __init__(self):
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
            self.w = Tensor.random_float(shape=(out_features, in_features), min=init_value_min, max=init_value_max)
            self.b = Tensor.random_float(shape=(out_features, ), min=init_value_min, max=init_value_max)

            self.dw = None
            self.db = None

        def forward(self, x: Tensor):
            # shapes x: (B, C_in) , w.T: (C_in, C_out)  b: (C_out)
            self.cache = x
            return x @ self.w.T + self.b

        def backward(self, dout: Tensor):
            x = self.cache

            # grad wrt weights
            self.dw = x.T @ dout

            # grad wrt bias
            self.db = dout.sum(dim=0)

            # grad wrt x
            dx = (dout @ self.w).reshape(x.shape)
            return dx

    class Sequential(Module):
        skip_cache = {}
        skip_grad_cache = {}

        def __init__(self, *modules):
            super().__init__()
            self.modules = modules

        def forward(self, x: Tensor):
            for module in self.modules:
                x = module(x)
            return x

        def backward(self, dout: Tensor):
            for module in reversed(self.modules):
                dout = module.backward(dout)
            return dout

    class SkipStart(Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x: Tensor):
            nn.Sequential.skip_cache[self.name] = x
            return x

        def backward(self, dout: Tensor):
            return dout + nn.Sequential.skip_cache[self.name]

    class SkipEnd(Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x: Tensor):
            return x + nn.Sequential.skip_cache.pop(self.name)

        def backward(self, dout: Tensor):
            nn.Sequential.skip_grad_cache[self.name] = dout
            return dout

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

    class Conv2d(Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
            super().__init__()
            init_value_min = - 0.1
            init_value_max = 0.1
            self.stride = stride
            self.padding = padding
            self.kernel_size = kernel_size
            self.out_channels = out_channels
            self.w = Tensor.random_float(shape=(out_channels, in_channels, kernel_size, kernel_size), min=init_value_min, max=init_value_max)
            self.b = Tensor.random_float(shape=(out_channels, ), min=init_value_min, max=init_value_max)

        def forward(self, x: Tensor):
            #  shapes x: (B, C_in, H, W)    w: (C_out, C_in, K, K)    b: (C_Out)    out: (B, C_out, H/s, W/s)
            B, C_in, H, W = x.shape
            C_out = self.out_channels
            K = self.kernel_size
            H_out = (H - 2 * self.padding) / self.stride
            W_out = (W - 2 * self.padding) / self.stride
            result = Tensor.zeros((B, C_out, H_out, W_out))
            for u in range(self.padding, H - self.padding, self.stride):
                for v in range(self.padding, H - self.padding, self.stride):
                    x_chunk = x[:, :, u:u+K, v:v+K].reshape((B,C_in, K*K))  # reshaped from B, C_in, K, K
                    result[:, :, u, v] = x_chunk @ self.w + self.b
                    #                   (B,C_in, K,K) @( C_out, C_in, K, K) + ( C_out)
                    # TODO complete

    # missing: weight_init, ConvTranspose2d, MaxPool2d, AvgPool2d, Softmax, BatchNorm2d, InstanceNorm2d, LayerNorm2d, Losses


if __name__ == "__main__":

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

    print("grad dx/dd", dout)

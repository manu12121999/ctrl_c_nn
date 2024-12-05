
def _f_transpose_2dim(matrix):
    # (I,J) -> (J,I)
    I, J = len(matrix), len(matrix[0])
    transposed = [[0 for _ in range(I)] for _ in range(J)]
    for i in range(I):
        for j in range(J):
            transposed[j][i] = matrix[i][j]
    return transposed


def _f_matmul_2dim(a, b):
    # (I,J) @ (J, K)  -> (I,K)
    I, J, J2, K = len(a), len(a[0]), len(b), len(b[0])
    assert J == J2
    result = [[0 for _ in range(K)] for _ in range(I)]

    # Transpose
    b_T = _f_transpose_2dim(b)

    # Perform matrix multiplication
    for i in range(I):
        for j in range(J):
            for k in range(K):
                result[i][j] += a[i][k] * b_T[j][k]

    return result


def _f_matmul(a, b):
    # input types: Tensors, output type: list
    if a.ndim == 2 and b.ndim == 2:
        return _f_matmul_2dim(a.elems, b.elems)
    elif a.nimds == 2 and b.ndim >= 3:
        return [_f_matmul(a, Tensor(b_i)) for b_i in b.elems]
    elif a.nimd >= 3 and b.ndim == 3:
        return [_f_matmul(Tensor(a_i), b) for a_i in a.elems]
    elif a.nimd >= 3 and b.ndim >= 3:
        return [_f_matmul(Tensor(a_i), Tensor(b_i)) for a_i, b_i in zip(a.elems, b.elems)]
    elif a.ndim < 2 or b.ndim < 2:
        raise NotImplementedError
import operator
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from ctrl_c_nn import Tensor, LLOps

matmul_shapes = [
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


add_shapes = [
    (100, 100),
    (1000, 1000),
    (1000, 25600),
    (25600, 1000),
]


def print_format(list):
    return [f"{x:.5f}" for x in list]


def plot_lists(**kwargs):
    plt.figure(figsize=(8, 6))
    for k, v in kwargs.items():
        if k == "title":
            plt.title(v)
        else:
            plt.plot(v, label=k, marker='o')

    plt.yscale('log')

    plt.xlabel("Index")
    plt.ylabel("Values (Log Scale)")

    plt.legend()


class BenchmarkTensor:

    def generic_various_shapes_test(self, func, init, shapes):
        # Testing various tensor shapes
        results = []

        for i in range(len(shapes)):
            a = init(np.random.randint(low=0, high=10, size=shapes[i]))
            b = init(np.random.randint(low=0, high=10, size=shapes[i]))

            time_start = time.time()
            result = func(a, b)
            time_end = time.time()

            results.append(time_end - time_start)
        return results

    def test_matmul_various_shapes(self):
        results_np = self.generic_various_shapes_test(np.matmul, lambda x: x, matmul_shapes)
        results_torch = self.generic_various_shapes_test(torch.matmul, lambda x: torch.tensor(x), matmul_shapes)
        results_ctrlc = self.generic_various_shapes_test(LLOps.f_matmul_2dim, lambda x: x.tolist(), matmul_shapes)

        print("numpy", print_format(results_np))
        print("torch", print_format(results_torch))
        print("ctrl_c", print_format(results_ctrlc))
        print("MATMUL slower than numpy",  np.array(results_ctrlc) / results_np, "on average", np.mean(np.array(results_ctrlc) / results_np))
        plot_lists(title="MATMUL", numpy=results_np, torch=results_torch, ctrl_c=results_ctrlc)
        plt.show()

    def test_add_various_shapes(self):
        results_np = self.generic_various_shapes_test(np.add, lambda x: x, add_shapes)
        results_torch = self.generic_various_shapes_test(torch.add, lambda x: torch.tensor(x), add_shapes)
        results_ctrlc = self.generic_various_shapes_test(lambda x, y: LLOps.f_operator(x,y,operator.add), lambda x: x.tolist(), add_shapes)

        print("numpy", print_format(results_np))
        print("torch", print_format(results_torch))
        print("ctrl_c", print_format(results_ctrlc))
        print("ADD slower than numpy",  np.array(results_ctrlc) / results_np, "on average", np.mean(np.array(results_ctrlc) / results_np))
        plot_lists(title="ADD", numpy=results_np, torch=results_torch, ctrl_c=results_ctrlc)


if __name__ == '__main__':
    a = BenchmarkTensor()
    a.test_matmul_various_shapes()
    a.test_add_various_shapes()

    plt.show()


    # MATMUL (python) numpy  ['0.00017', '0.00100', '0.01168', '0.12945']
    # MATMUL (cython) ctrl_c ['0.02198', '0.14381', '0.99213', '9.48446']
    # MATMUL (pypy)   ctrl_c ['0.01763', '0.03212', '0.19283', '1.37085']

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from ctrl_c_nn import _f_matmul_2dim

shapes = [
    (16, 16),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]


def plot_lists(**kwargs):
    plt.figure(figsize=(8, 6))
    for k, v in kwargs.items():
        plt.plot(v, label=k, marker='o')

    plt.yscale('log')

    plt.title("Logarithmic Scale Plot")
    plt.xlabel("Index")
    plt.ylabel("Values (Log Scale)")

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()


class BenchmarkTensor:

    def generic_various_shapes_test(self, func1, func2, func3):
        # Testing various tensor shapes
        list_a = []
        list_b = []
        list_c = []

        for i in range(len(shapes)):
            a_np = np.random.randint(low=0, high=10, size=shapes[i])
            b_np = np.random.randint(low=0, high=10, size=shapes[i])
            a_torch, b_torch = torch.tensor(a_np), torch.tensor(b_np)
            a_ctrl_c, b_ctrl_c = a_np.tolist(), b_np.tolist()

            time_start = time.time()
            result_np = func1(a_np, b_np)
            result_np[0, 0] = result_np[0, 0]
            time_end = time.time()

            time_start2 = time.time()
            result_torch = func2(a_torch, b_torch)
            result_torch[0, 0] = result_torch[0, 0]
            time_end2 = time.time()

            time_start3 = time.time()
            result_torch = func3(a_ctrl_c, b_ctrl_c)
            result_torch[0][0] = result_torch[0][0]
            time_end3 = time.time()

            list_a.append(time_end - time_start)
            list_b.append(time_end2 - time_start2)
            list_c.append(time_end3 - time_start3)
        return list_a, list_b, list_c

    def test_matmul_various_shapes(self):
        list_a, list_b, list_c = self.generic_various_shapes_test(np.matmul, torch.matmul, _f_matmul_2dim)
        print("numpy", list_a)
        print("torch", list_b)
        print("ctrl_c", list_c)
        plot_lists(numpy=list_a, torch=list_b, ctrl_c=list_c)


    def test_add_various_shapes(self):
        list_a, list_b, list_c = self.generic_various_shapes_test(np.add, torch.add, _f_matmul_2dim)
        print("numpy", list_a)
        print("torch", list_b)
        print("ctrl_c", list_c)


if __name__ == '__main__':
    a = BenchmarkTensor()
    a.test_matmul_various_shapes()
    #a.test_add_various_shapes()
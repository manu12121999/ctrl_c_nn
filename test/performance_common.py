import sys
import cProfile
import time
from pathlib import Path

sys.path.append(str(Path('.').absolute().parent))

# import torch
from ctrl_c_nn import Tensor, nn

x = Tensor.random_float(shape=(16, 256))

model = nn.Sequential(
                nn.SkipStart("a"),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.SkipStart("b"),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.SkipEnd("b"),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.SkipEnd("a"),
                nn.Linear(256, 3)
    )


# x_torch = torch.rand((16, 256))
# model_pytorch = torch.nn.Sequential(
#         torch.nn.Linear(256, 256),
#         torch.nn.ReLU(),
#         torch.nn.Linear(256, 512),
#         torch.nn.LeakyReLU(),
#         torch.nn.Linear(512, 4),
# )

t0 = time.time()
res = model(x)
t1 = time.time()
print(f"ctrl_c took {t1-t0:.4f} s")

# t0 = time.time()
# res = model_pytorch(x_torch)
# t1 = time.time()
#
# print(f"pytorch took {t1-t0:.4f} s")


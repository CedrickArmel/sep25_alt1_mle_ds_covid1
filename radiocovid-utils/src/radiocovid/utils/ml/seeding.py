# MIT License
#
# Copyright (c) 2025 @CedrickArmel, @samarita22, @TaxelleT & @Yeyecodes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import random

import numpy as np
import torch


def get_seeded_generator(seed: "int") -> "torch.Generator":
    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1
    seed = seed % MAX_SEED
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(
    seed: "int | None" = 4294967295,
    cudnn_backend: "bool" = False,
    use_deterministic_algorithms: "bool" = False,
    warn_only: "bool" = True,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1

    if seed is None:
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    if seed is not None and cudnn_backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(
            mode=use_deterministic_algorithms, warn_only=warn_only
        )

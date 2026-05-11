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

import random

import torch

from ..pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def worker_balanced_n_samples(n_samples: int, batch: int, worldsize: int):
    if (x := n_samples % (batch * worldsize)) != 0:
        n = n_samples + ((batch * worldsize) - x)
        return n
    else:
        return n_samples


def balance_data_world_size(data: list[dict], batch: int, worldsize: int) -> list[dict]:
    """Complete `data` by randomly sampling a set of observations from data.
    Useful to avoid `drop_last` in data loader and train/evaluate on whole data.

    Args:
        data (list[dict]): The data to complete.
        batch (int): Batch size.
        worldsize (int): Worldsize

    Returns:
        list[dict]: Completed data
    """
    if (x := len(data) % (batch * worldsize)) != 0:
        data += random.choices(population=data, k=((batch * worldsize) - x))
    return data


def get_process_group():
    if not torch.distributed.is_initialized():
        return None
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    if device_type == "cuda":
        backend = "nccl"
        log.info("Instanciating a NCCL backend...")
    elif device_type == "cpu":
        backend = "gloo"
        log.info("Instanciating a GLOO backend...")
    else:
        return None
    return torch.distributed.new_group(backend=backend)

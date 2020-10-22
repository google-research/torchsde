# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test `BrownianTree`.

The suite tests both running on CPU and CUDA (if available).
"""
import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import numpy as np
import numpy.random as npr
import torch
from scipy.stats import norm, kstest

import pytest
import torchsde

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

D = 3
SMALL_BATCH_SIZE = 16
LARGE_BATCH_SIZE = 16384
REPS = 3
ALPHA = 0.00001

devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]


def _setup(device, batch_size):
    t0, t1 = torch.tensor([0., 1.], device=device)
    w0 = torch.zeros(batch_size, D, device=device)
    t = torch.rand([]).to(device)
    bm = torchsde.BrownianTree(t0=t0, t1=t1, w0=w0, entropy=0)
    return t, bm


def _dict_to_sorted_list(*dicts):
    lists = tuple([d[k] for k in sorted(d.keys())] for d in dicts)
    if len(lists) == 1:
        return lists[0]
    return lists


@pytest.mark.parametrize("device", devices)
def test_basic(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(device, SMALL_BATCH_SIZE)
    sample = bm(t)
    assert sample.size() == (SMALL_BATCH_SIZE, D)


@pytest.mark.parametrize("device", devices)
def test_determinism(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(device, SMALL_BATCH_SIZE)
    vals = [bm(t) for _ in range(REPS)]
    for val in vals[1:]:
        assert torch.allclose(val, vals[0])


@pytest.mark.parametrize("device", devices)
def test_normality(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0_, t1_ = 0.0, 1.0
    t0, t1 = torch.tensor([t0_, t1_], device=device)
    eps = 1e-5
    for _ in range(REPS):
        w0_, w1_ = 0.0, npr.randn()
        w0 = torch.tensor(w0_, device=device).repeat(LARGE_BATCH_SIZE)
        w1 = torch.tensor(w1_, device=device).repeat(LARGE_BATCH_SIZE)
        bm = torchsde.BrownianTree(t0=t0, t1=t1, w0=w0, w1=w1, pool_size=100, tol=1e-14)  # noqa

        for _ in range(REPS):
            t_ = npr.uniform(low=t0_ + eps, high=t1_ - eps)
            samples = bm(t_)
            samples_ = samples.cpu().detach().numpy()

            mean_ = ((t1_ - t_) * w0_ + (t_ - t0_) * w1_) / (t1_ - t0_)
            std_ = np.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
            ref_dist = norm(loc=mean_, scale=std_)

            _, pval = kstest(samples_, ref_dist.cdf)
            assert pval >= ALPHA

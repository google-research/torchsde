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

"""Test `BrownianPath`.

The suite tests both running on CPU and CUDA (if available).
"""
import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import math
import numpy.random as npr
import torch
from scipy.stats import norm, kstest

import itertools

import torchsde
import pytest

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)

D = 3
BATCH_SIZE = 131072
REPS = 3
ALPHA = 0.001

brownian_classes = [torchsde.brownian_lib.BrownianPath, torchsde.BrownianPath]
devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]


def _setup(brownian_class, device):
    t0, t1 = torch.tensor([0., 1.], device=device)
    w0, w1 = torch.randn([2, BATCH_SIZE, D], device=device)
    t = torch.rand([], device=device)
    bm = brownian_class(t0=t0, w0=w0)
    return t, bm


@pytest.mark.parametrize("brownian_class, device", itertools.product(brownian_classes, devices))
def test_basic(brownian_class, device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(brownian_class, device)
    sample = bm(t)
    assert sample.size() == (BATCH_SIZE, D)


@pytest.mark.parametrize("brownian_class, device", itertools.product(brownian_classes, devices))
def test_determinism(brownian_class, device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(brownian_class, device)
    vals = [bm(t) for _ in range(REPS)]
    for val in vals[1:]:
        assert torch.allclose(val, vals[0])


@pytest.mark.parametrize("brownian_class", brownian_classes)
@pytest.mark.parametrize("device", devices)
def test_normality(brownian_class, device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0_, t1_ = 0.0, 1.0
    t0, t1 = torch.tensor([t0_, t1_], device=device)
    eps = 1e-2
    for _ in range(REPS):
        w0_, w1_ = 0.0, npr.randn() * math.sqrt(t1_)
        w0 = torch.tensor(w0_, device=device).repeat(BATCH_SIZE)
        w1 = torch.tensor(w1_, device=device).repeat(BATCH_SIZE)

        bm = brownian_class(t0=t0, w0=w0)  # noqa
        bm.insert(t=t1, w=w1)

        t_ = npr.uniform(low=t0_ + eps, high=t1_ - eps)  # Avoid sampling too close to the boundary.
        samples = bm(t_)
        samples_ = samples.cpu().detach().numpy()

        # True expected mean from Brownian bridge.
        mean_ = ((t1_ - t_) * w0_ + (t_ - t0_) * w1_) / (t1_ - t0_)
        std_ = math.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
        ref_dist = norm(loc=mean_, scale=std_)

        _, pval = kstest(samples_, ref_dist.cdf)
        assert pval >= ALPHA


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("random_order", [False, True])
def test_continuity(device, random_order):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ts = torch.linspace(0., 1., 10000, device=device)
    bm = torchsde.BrownianPath(t0=ts[0], t1=ts[-1], w0=torch.randn((), dtype=ts.dtype, device=device))
    vals = torch.empty_like(ts)
    t_indices = torch.randperm(len(ts), device=device) if random_order else torch.arange(len(ts), device=device)
    for t_index in t_indices:
        t = ts[t_index]
        vals[t_index] = bm(t)
    last_val = vals[0]
    for val in vals[1:]:
        assert (val - last_val).abs().max() < 5e-2
        last_val = val

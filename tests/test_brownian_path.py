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
import numpy as np
import numpy.random as npr
import torch
from scipy.stats import norm, kstest

import torchsde
import pytest

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)

D = 3
BATCH_SIZE = 131072
REPS = 3
ALPHA = 0.00001

devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]


def _setup(device):
    t0, t1 = torch.tensor([0., 1.], device=device)
    w0, w1 = torch.randn([2, BATCH_SIZE, D], device=device)
    t = torch.rand([], device=device)
    bm = torchsde.BrownianPath(t0=t0, w0=w0)
    return t, bm


@pytest.mark.parametrize("device", devices)
def test_basic(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(device)
    sample = bm(t)
    assert sample.size() == (BATCH_SIZE, D)


@pytest.mark.parametrize("device", devices)
def test_determinism(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(device)
    vals = [bm(t) for _ in range(REPS)]
    for val in vals[1:]:
        assert torch.allclose(val, vals[0])


@pytest.mark.parametrize("device", devices)
def test_normality(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0_, t1_ = 0.0, 1.0
    eps = 1e-2
    for _ in range(REPS):
        w0_ = npr.randn() * math.sqrt(t1_)
        w0 = torch.tensor(w0_, device=device).repeat(BATCH_SIZE)

        bm = torchsde.BrownianPath(t0=t0_, w0=w0)  # noqa

        w1_ = bm(t1_).cpu().numpy()

        t_ = npr.uniform(low=t0_ + eps, high=t1_ - eps)  # Avoid sampling too close to the boundary.
        samples_ = bm(t_).cpu().numpy()

        # True expected mean from Brownian bridge.
        mean_ = ((t1_ - t_) * w0_ + (t_ - t0_) * w1_) / (t1_ - t0_)
        std_ = math.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
        ref_dist = norm(loc=np.zeros_like(mean_), scale=np.ones_like(std_))

        _, pval = kstest((samples_ - mean_) / std_, ref_dist.cdf)
        assert pval >= ALPHA

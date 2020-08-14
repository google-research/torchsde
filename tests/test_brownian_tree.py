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

"""Test the two `BrownianTree`.

The suite tests both running on CPU and CUDA (if available).
"""
import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import numpy as np
import numpy.random as npr
import torch
from scipy.stats import norm, kstest

import itertools
import pytest
import torchsde

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

D = 3
SMALL_BATCH_SIZE = 16
LARGE_BATCH_SIZE = 16384
REPS = 3
ALPHA = 0.001

brownian_classes = [torchsde.brownian_lib.BrownianTree, torchsde.BrownianTree]
devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]


def _setup(brownian_class, device, batch_size):
    t0, t1 = torch.tensor([0., 1.]).to(device)
    w0 = torch.zeros(batch_size, D).to(device)
    t = torch.rand([]).to(device)
    bm = brownian_class(t0=t0, t1=t1, w0=w0, entropy=0)
    return t, bm


def _dict_to_sorted_list(*dicts):
    lists = tuple([d[k] for k in sorted(d.keys())] for d in dicts)
    if len(lists) == 1:
        return lists[0]
    return lists


@pytest.mark.parametrize("brownian_class, device", itertools.product(brownian_classes, devices))
def test_basic(brownian_class, device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(brownian_class, device, SMALL_BATCH_SIZE)
    sample = bm(t)
    assert sample.size() == (SMALL_BATCH_SIZE, D)


@pytest.mark.parametrize("brownian_class, device", itertools.product(brownian_classes, devices))
def test_determinism(brownian_class, device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(brownian_class, device, SMALL_BATCH_SIZE)
    vals = [bm(t) for _ in range(REPS)]
    for val in vals[1:]:
        assert torch.allclose(val, vals[0])


@pytest.mark.parametrize("brownian_class, device", itertools.product(brownian_classes, devices))
def test_normality(brownian_class, device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0_, t1_ = 0.0, 1.0
    t0, t1 = torch.tensor([t0_, t1_]).to(device)
    eps = 1e-5
    for _ in range(REPS):
        w0_, w1_ = 0.0, npr.randn()
        w0 = torch.tensor(w0_).repeat(LARGE_BATCH_SIZE).to(device)
        w1 = torch.tensor(w1_).repeat(LARGE_BATCH_SIZE).to(device)
        bm = brownian_class(t0=t0, t1=t1, w0=w0, w1=w1, pool_size=100, tol=1e-14)  # noqa

        for _ in range(REPS):
            t_ = npr.uniform(low=t0_ + eps, high=t1_ - eps)
            samples = bm(t_)
            samples_ = samples.cpu().detach().numpy()

            mean_ = ((t1_ - t_) * w0_ + (t_ - t0_) * w1_) / (t1_ - t0_)
            std_ = np.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
            ref_dist = norm(loc=mean_, scale=std_)

            _, pval = kstest(samples_, ref_dist.cdf)
            assert pval >= ALPHA


@pytest.mark.parametrize("brownian_class, device", itertools.product(brownian_classes, devices))
def test_to_float32(brownian_class, device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(brownian_class, device, SMALL_BATCH_SIZE)
    curr, prev, post = _dict_to_sorted_list(*bm.get_cache())
    old = torch.cat(curr + prev + post, dim=0)

    bm.to(torch.float32)
    curr, prev, post = _dict_to_sorted_list(*bm.get_cache())
    new = torch.cat(curr + prev + post, dim=0)
    assert new.dtype == torch.float32
    assert torch.allclose(old, new.double())


@pytest.mark.parametrize("brownian_class", brownian_classes)
def test_to_device(brownian_class):
    if not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t, bm = _setup(brownian_class, cpu, SMALL_BATCH_SIZE)
    curr, prev, post = _dict_to_sorted_list(*bm.get_cache())
    old = torch.cat(curr + prev + post, dim=0)

    bm.to(gpu)
    curr, prev, post = _dict_to_sorted_list(*bm.get_cache())
    new = torch.cat(curr + prev + post, dim=0)
    assert str(new.device).startswith('cuda')
    assert torch.allclose(old, new.cpu())

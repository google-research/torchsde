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

import pytest
import torchsde

torch.manual_seed(2147483647)
torch.set_default_dtype(torch.float64)

D = 3
SMALL_BATCH_SIZE = 16
LARGE_BATCH_SIZE = 16384
REPS = 3
LARGE_REPS = 500
ALPHA = 0.001


devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]


def _setup(device, batch_size):
    t0, t1 = torch.tensor([0., 1.], device=device)
    ta = torch.rand([], device=device)
    tb = torch.rand([], device=device)
    ta, tb = min(ta, tb), max(ta, tb)
    bm = torchsde.BrownianInterval(t0=t0, t1=t1, shape=(batch_size, D), dtype=torch.float64, device=device)
    return ta, tb, bm


@pytest.mark.parametrize("device", devices)
def test_shape(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(device, SMALL_BATCH_SIZE)
    sample1 = bm(ta)
    sample2 = bm(tb)
    sample3 = bm(ta, tb)
    assert sample1.shape == sample2.shape == sample3.shape == (SMALL_BATCH_SIZE, D)

    ta, tb, bm = _setup(device, SMALL_BATCH_SIZE)
    # Query interval before increment
    sample3 = bm(ta, tb)
    sample1 = bm(ta)
    sample2 = bm(tb)
    assert sample1.shape == sample2.shape == sample3.shape == (SMALL_BATCH_SIZE, D)


@pytest.mark.parametrize("device", devices)
def test_determinism_simple(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(device, SMALL_BATCH_SIZE)
    vals = [bm(ta, tb) for _ in range(REPS)]
    for val in vals[1:]:
        assert torch.allclose(val, vals[0])


@pytest.mark.parametrize("device", devices)
def test_determinism_large(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(device, SMALL_BATCH_SIZE)
    cache = {}
    for _ in range(LARGE_REPS):
        ta_ = torch.rand_like(ta)
        tb_ = torch.rand_like(tb)
        ta_, tb_ = min(ta_, tb_), max(ta_, tb_)
        val = bm(ta_, tb_)
        cache[ta_, tb_] = val.detach().clone()

    cache2 = {}
    for ta_, tb_ in cache:
        val = bm(ta_, tb_)
        cache2[ta_, tb_] = val.detach().clone()

    for ta_, tb_ in cache:
        assert (cache[ta_, tb_] == cache2[ta_, tb_]).all()


@pytest.mark.parametrize("device", devices)
def test_normality_simple(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0_, t1_ = 0.0, 1.0
    t0, t1 = torch.tensor([t0_, t1_], device=device)
    eps = 1e-5
    for _ in range(REPS):
        W = torch.tensor(npr.randn(), device=device).repeat(LARGE_BATCH_SIZE)
        bm = torchsde.BrownianInterval(t0=t0, t1=t1, W=W)

        for _ in range(REPS):
            t_ = npr.uniform(low=t0_ + eps, high=t1_ - eps)
            samples = bm(t_)
            samples_ = samples.cpu().detach().numpy()

            mean_ = W.cpu() * (t_ - t0_) / (t1_ - t0_)
            std_ = np.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
            ref_dist = norm(loc=mean_, scale=std_)

            _, pval = kstest(samples_, ref_dist.cdf)
            assert pval >= ALPHA


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("random_order", [False, True])
def test_continuity(device, random_order):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ts = torch.linspace(0., 1., 10000, device=device)
    bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], shape=(), dtype=ts.dtype, device=device)
    vals = torch.empty_like(ts)
    i_ = torch.arange(len(ts), device=device)
    if random_order:
        i_ = i_[torch.randperm(len(ts), device=device)]
    for i in i_:
        t = ts[i]
        vals[i] = bm(t)
    last_val = vals[0]
    for val in vals[1:]:
        assert (val - last_val).abs().max() < 5e-2
        last_val = val


@pytest.mark.parametrize("device", devices)
def test_to_dtype(device):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(device, SMALL_BATCH_SIZE)
    tc = torch.rand_like(ta)
    td = torch.rand_like(ta)
    tc, td = min(tc, td), max(tc, td)
    te = torch.rand_like(ta)
    tf = torch.rand_like(ta)
    te, tf = min(te, tf), max(te, tf)
    tg = torch.rand_like(ta)
    th = torch.rand_like(ta)
    tg, th = min(tg, th), max(tg, th)

    w = bm(ta, tb)
    w2 = bm(tc, td)

    bm.to(torch.float32)
    w_ = bm(ta, tb)
    w2_ = bm(tc, td)
    w3_ = bm(te, tf)
    w4_ = bm(tg, th)

    bm.to(torch.float64)
    w3 = bm(te, tf)
    w4 = bm(tg, th)

    assert w.dtype == w2.dtype == w3.dtype == w4.dtype == torch.float64
    assert w_.dtype == w2_.dtype == w3_.dtype == w4_.dtype == torch.float32
    assert torch.allclose(w, w_.double())
    assert torch.allclose(w2, w2_.double())
    assert torch.allclose(w3, w3_.double())
    assert torch.allclose(w4, w4_.double())


def test_to_device():
    if not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(cpu, SMALL_BATCH_SIZE)
    tc = torch.rand_like(ta)
    td = torch.rand_like(ta)
    tc, td = min(tc, td), max(tc, td)
    te = torch.rand_like(ta)
    tf = torch.rand_like(ta)
    te, tf = min(te, tf), max(te, tf)
    tg = torch.rand_like(ta)
    th = torch.rand_like(ta)
    tg, th = min(tg, th), max(tg, th)

    w = bm(ta, tb)
    w2 = bm(tc, td)

    bm.to(gpu)
    w_ = bm(ta, tb)
    w2_ = bm(tc, td)
    w3_ = bm(te, tf)
    w4_ = bm(tg, th)

    bm.to(cpu)
    w3 = bm(te, tf)
    w4 = bm(tg, th)

    assert str(w.device).startswith('cpu')
    assert str(w2.device).startswith('cpu')
    assert str(w3.device).startswith('cpu')
    assert str(w4.device).startswith('cpu')
    assert str(w_.device).startswith('cuda')
    assert str(w2_.device).startswith('cuda')
    assert str(w3_.device).startswith('cuda')
    assert str(w4_.device).startswith('cuda')
    assert torch.allclose(w, w_.cpu())
    assert torch.allclose(w2, w2_.cpu())
    assert torch.allclose(w3, w3_.cpu())
    assert torch.allclose(w4, w4_.cpu())

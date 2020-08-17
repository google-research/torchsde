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

"""Test the `BrownianInterval``.

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

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)

D = 3
SMALL_BATCH_SIZE = 16
LARGE_BATCH_SIZE = 32000
REPS = 3
LARGE_REPS = 500
ALPHA = 0.001


devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]


def _setup(device, levy_area_approximation, shape):
    t0, t1 = torch.tensor([0., 1.], device=device)
    ta = torch.rand([], device=device)
    tb = torch.rand([], device=device)
    ta, tb = min(ta, tb), max(ta, tb)
    bm = torchsde.BrownianInterval(t0=t0, t1=t1, shape=shape, device=device,
                                   levy_area_approximation=levy_area_approximation)
    return ta, tb, bm


def _levy_returns():
    yield "none", False, False
    yield "space-time", False, False
    yield "space-time", True, False
    for levy_area_approximation in ('davie', 'foster'):
        for return_U in (True, False):
            for return_A in (True, False):
                yield levy_area_approximation, return_U, return_A


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation, return_U, return_A", _levy_returns())
def test_shape(device, levy_area_approximation, return_U, return_A):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    for shape, A_shape in (((SMALL_BATCH_SIZE, D), (SMALL_BATCH_SIZE, D, D)),
                           ((SMALL_BATCH_SIZE,), (SMALL_BATCH_SIZE,)),
                           ((), ())):
        ta, tb, bm = _setup(device, levy_area_approximation, shape)
        sample1 = bm(ta, return_U=return_U, return_A=return_A)
        sample2 = bm(tb, return_U=return_U, return_A=return_A)
        sample3 = bm(ta, tb, return_U=return_U, return_A=return_A)
        shapes = []
        A_shapes = []
        for sample in (sample1, sample2, sample3):
            if return_U:
                if return_A:
                    W1, U1, A1 = sample
                    shapes.append(W1.shape)
                    shapes.append(U1.shape)
                    A_shapes.append(A1.shape)
                else:
                    W1, U1 = sample
                    shapes.append(W1.shape)
                    shapes.append(U1.shape)
            else:
                if return_A:
                    W1, A1 = sample
                    shapes.append(W1.shape)
                    A_shapes.append(A1.shape)
                else:
                    W1 = sample
                    shapes.append(W1.shape)

        for shape_ in shapes:
            assert shape_ == shape
        for shape_ in A_shapes:
            assert shape_ == A_shape


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation, return_U, return_A", _levy_returns())
def test_determinism_simple(device, levy_area_approximation, return_U, return_A):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(device, levy_area_approximation, (SMALL_BATCH_SIZE, D))
    vals = [bm(ta, tb, return_U=return_U, return_A=return_A) for _ in range(REPS)]
    for val in vals[1:]:
        if torch.is_tensor(val):
            val = (val,)
        if torch.is_tensor(vals[0]):
            val0 = (vals[0],)
        else:
            val0 = vals[0]
        for v, v0 in zip(val, val0):
            assert (v == v0).all()


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation, return_U, return_A", _levy_returns())
def test_determinism_large(device, levy_area_approximation, return_U, return_A):
    """
    Tests that BrownianInterval deterministically produces the same results when queried at the same points.

    We first of all query it at lots of points (larger than its internal cache), and then re-query at the same set of
    points, and compare.
    """
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(device, levy_area_approximation, (SMALL_BATCH_SIZE, D))
    cache = {}
    for _ in range(LARGE_REPS):
        ta_ = torch.rand_like(ta)
        tb_ = torch.rand_like(tb)
        ta_, tb_ = min(ta_, tb_), max(ta_, tb_)
        val = bm(ta_, tb_, return_U=return_U, return_A=return_A)
        if torch.is_tensor(val):
            val = (val,)
        cache[ta_, tb_] = tuple(v.detach().clone() for v in val)

    cache2 = {}
    for ta_, tb_ in cache:
        val = bm(ta_, tb_, return_U=return_U, return_A=return_A)
        if torch.is_tensor(val):
            val = (val,)
        cache2[ta_, tb_] = tuple(v.detach().clone() for v in val)

    for ta_, tb_ in cache:
        for v1, v2 in zip(cache[ta_, tb_], cache2[ta_, tb_]):
            assert (v1 == v2).all()


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation", ['none', 'space-time', 'davie', 'foster'])
def test_normality_simple(device, levy_area_approximation):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0_, t1_ = 0.0, 1.0
    t0, t1 = torch.tensor([t0_, t1_], device=device)
    for _ in range(REPS):
        W = torch.tensor(npr.randn(), device=device).repeat(LARGE_BATCH_SIZE)
        bm = torchsde.BrownianInterval(t0=t0, t1=t1, W=W, levy_area_approximation=levy_area_approximation)

        for _ in range(REPS):
            t_ = npr.uniform(low=t0_, high=t1_)
            samples = bm(t_)
            samples_ = samples.cpu().detach().numpy()

            mean_ = W.cpu() * (t_ - t0_) / (t1_ - t0_)
            std_ = np.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
            ref_dist = norm(loc=mean_, scale=std_)

            _, pval = kstest(samples_, ref_dist.cdf)
            assert pval >= ALPHA


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation", ['none', 'space-time', 'davie', 'foster'])
def test_normality_conditional(device, levy_area_approximation):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0_, t1_ = 0.0, 1.0
    t0, t1 = torch.tensor([t0_, t1_], device=device)
    for _ in range(REPS):
        bm = torchsde.BrownianInterval(t0=t0, t1=t1, shape=(LARGE_BATCH_SIZE,),
                                       levy_area_approximation=levy_area_approximation)

        for _ in range(REPS):
            t_ = npr.uniform(low=t0_, high=t1_)
            ta = npr.uniform(low=t0_, high=t1_)
            tb = npr.uniform(low=t0_, high=t1_)
            ta, t_, tb = sorted([t_, ta, tb])

            increment = bm(ta, tb).cpu().detach().numpy()
            sample = bm(ta, t_).cpu().detach().numpy()

            mean = increment * (t_ - ta) / (tb - ta)
            std = np.sqrt((tb - t_) * (t_ - ta) / (tb - ta))
            rescaled_sample = (sample - mean) / std

            _, pval = kstest(rescaled_sample, 'norm')
            assert pval >= ALPHA


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation", ['none', 'space-time', 'davie', 'foster'])
@pytest.mark.parametrize("random_order", [False, True])
def test_continuity(device, levy_area_approximation, random_order):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ts = torch.linspace(0., 1., 10000, device=device)
    bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], shape=(), device=device,
                                   levy_area_approximation=levy_area_approximation)
    vals = torch.empty_like(ts)
    t_indices = torch.randperm(len(ts), device=device) if random_order else torch.arange(len(ts), device=device)
    for t_index in t_indices:
        t = ts[t_index]
        vals[t_index] = bm(t)
    last_val = vals[0]
    for val in vals[1:]:
        assert (val - last_val).abs().max() < 5e-2
        last_val = val


@pytest.mark.skip("BrownianInterval.to not supported")
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation", ['none', 'space-time', 'davie', 'foster'])
def test_to_dtype(device, levy_area_approximation):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(device, levy_area_approximation, (SMALL_BATCH_SIZE, D))
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


@pytest.mark.skip("BrownianInterval.to not supported")
@pytest.mark.parametrize("levy_area_approximation", ['none', 'space-time', 'davie', 'foster'])
def test_to_device(levy_area_approximation):
    if not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    ta, tb, bm = _setup(cpu, levy_area_approximation, (SMALL_BATCH_SIZE, D))
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

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

"""Test `BrownianInterval`.

The suite tests both running on CPU and CUDA (if available).
"""
import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import math
import numpy.random as npr
import torch
from scipy.stats import kstest

import pytest
import torchsde
from torchsde._brownian import utils

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)

D = 3
SMALL_BATCH_SIZE = 16
LARGE_BATCH_SIZE = 131072
REPS = 3
MEDIUM_REPS = 25
LARGE_REPS = 500
ALPHA = 0.0001
POOL_SIZE = 32

devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]


def _setup(device, levy_area_approximation, shape):
    t0, t1 = torch.tensor([0., 1.], device=device)
    ta = torch.rand([], device=device)
    tb = torch.rand([], device=device)
    ta, tb = min(ta, tb), max(ta, tb)
    bm = torchsde.BrownianInterval(t0=t0, t1=t1, shape=shape, device=device,
                                   levy_area_approximation=levy_area_approximation, pool_size=POOL_SIZE)
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

    t0, t1 = 0.0, 1.0
    for _ in range(REPS):
        base_W = torch.tensor(npr.randn(), device=device).repeat(LARGE_BATCH_SIZE)
        bm = torchsde.BrownianInterval(t0=t0, t1=t1, W=base_W, levy_area_approximation=levy_area_approximation)

        t_ = npr.uniform(low=t0, high=t1)

        W = bm(t0, t_)

        mean_W = base_W * (t_ - t0) / (t1 - t0)
        std_W = math.sqrt((t1 - t_) * (t_ - t0) / (t1 - t0))
        rescaled_W = (W - mean_W) / std_W

        _, pval = kstest(rescaled_W.cpu().detach().numpy(), 'norm')
        assert pval >= ALPHA

        if levy_area_approximation != 'none':
            W, U = bm(t0, t_, return_U=True)
            H = utils.U_to_H(W, U, t_ - t0)

            mean_H = 0
            std_H = math.sqrt((t_ - t0) / 12)
            rescaled_H = (H - mean_H) / std_H

            _, pval = kstest(rescaled_H.cpu().detach().numpy(), 'norm')
            assert pval >= ALPHA


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation", ['none', 'space-time', 'davie', 'foster'])
def test_normality_conditional(device, levy_area_approximation):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0, t1 = 0.0, 1.0
    for _ in range(REPS):
        bm = torchsde.BrownianInterval(t0=t0, t1=t1, shape=(LARGE_BATCH_SIZE,), device=device,
                                       levy_area_approximation=levy_area_approximation, pool_size=POOL_SIZE)

        for _ in range(MEDIUM_REPS):
            ta, t_, tb = sorted(npr.uniform(low=t0, high=t1, size=(3,)))

            W = bm(ta, tb)
            W1 = bm(ta, t_)
            W2 = bm(t_, tb)

            mean_W1 = W * (t_ - ta) / (tb - ta)
            std_W1 = math.sqrt((tb - t_) * (t_ - ta) / (tb - ta))
            rescaled_W1 = (W1 - mean_W1) / std_W1
            _, pval = kstest(rescaled_W1.cpu().detach().numpy(), 'norm')
            assert pval >= ALPHA

            mean_W2 = W * (tb - t_) / (tb - ta)
            std_W2 = math.sqrt((tb - t_) * (t_ - ta) / (tb - ta))
            rescaled_W2 = (W2 - mean_W2) / std_W2
            _, pval = kstest(rescaled_W2.cpu().detach().numpy(), 'norm')
            assert pval >= ALPHA

            if levy_area_approximation != 'none':
                W, U = bm(ta, tb, return_U=True)
                W1, U1 = bm(ta, t_, return_U=True)
                W2, U2 = bm(t_, tb, return_U=True)

                h = tb - ta
                h1 = t_ - ta
                h2 = tb - t_

                denom = math.sqrt(h1 ** 3 + h2 ** 3)
                a = h1 ** 3.5 * h2 ** 0.5 / (2 * h * denom)
                b = h1 ** 0.5 * h2 ** 3.5 / (2 * h * denom)
                c = math.sqrt(3) * h1 ** 1.5 * h2 ** 1.5 / (6 * denom)

                H = utils.U_to_H(W, U, h)
                H1 = utils.U_to_H(W1, U1, h1)
                H2 = utils.U_to_H(W2, U2, h2)

                mean_H1 = H * (h1 / h) ** 2
                std_H1 = math.sqrt(a**2 + c**2) / h1
                rescaled_H1 = (H1 - mean_H1) / std_H1

                _, pval = kstest(rescaled_H1.cpu().detach().numpy(), 'norm')
                assert pval >= ALPHA

                mean_H2 = H * (h2 / h) ** 2
                std_H2 = math.sqrt(b**2 + c**2) / h2
                rescaled_H2 = (H2 - mean_H2) / std_H2

                _, pval = kstest(rescaled_H2.cpu().detach().numpy(), 'norm')
                assert pval >= ALPHA


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("levy_area_approximation", ['none', 'space-time', 'davie', 'foster'])
def test_consistency(device, levy_area_approximation):
    if device == gpu and not torch.cuda.is_available():
        pytest.skip(msg="CUDA not available.")

    t0, t1 = 0.0, 1.0
    for _ in range(REPS):
        bm = torchsde.BrownianInterval(t0=t0, t1=t1, shape=(LARGE_BATCH_SIZE,),
                                       device=device,
                                       levy_area_approximation=levy_area_approximation, pool_size=POOL_SIZE)

        for _ in range(MEDIUM_REPS):
            ta, t_, tb = sorted(npr.uniform(low=t0, high=t1, size=(3,)))

            if levy_area_approximation == 'none':
                W = bm(ta, tb)
                W1 = bm(ta, t_)
                W2 = bm(t_, tb)
            else:
                W, U = bm(ta, tb, return_U=True)
                W1, U1 = bm(ta, t_, return_U=True)
                W2, U2 = bm(t_, tb, return_U=True)

            torch.testing.assert_allclose(W1 + W2, W, rtol=1e-6, atol=1e-6)
            if levy_area_approximation != 'none':
                torch.testing.assert_allclose(U1 + U2 + (tb - t_) * W1, U, rtol=1e-6, atol=1e-6)

            # We don't test the return_A case because we don't expect that to be consistent.

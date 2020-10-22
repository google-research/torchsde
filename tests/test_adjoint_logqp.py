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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import unittest

import torch

from tests.basic_sde import BasicSDE1, BasicSDE2, BasicSDE3, BasicSDE4
from tests.torch_test import TorchTestCase
from torchsde import BrownianPath, sdeint_adjoint

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d = 2
m = 2
batch_size = 1
t0, t1, t2 = ts = torch.tensor([0.0, 0.15, 0.3]).to(device)
dt = 1e-3
w0 = torch.zeros(batch_size, d).to(device)
y0 = torch.zeros(batch_size, d).to(device).fill_(0.1)
methods = ('milstein', 'srk')


class TestAdjointLogqp(TorchTestCase):

    def test_basic_sde1(self):
        sde = BasicSDE1(d).to(device)
        _test_forward_and_backward(sde)

    def test_basic_sde2(self):
        sde = BasicSDE2(d).to(device)
        _test_forward_and_backward(sde)

    def test_basic_sde3(self):
        sde = BasicSDE3(d).to(device)
        _test_forward_and_backward(sde)

    def test_basic_sde4(self):
        sde = BasicSDE4(d).to(device)
        _test_forward_and_backward(sde)


def _test_forward_and_backward(sde):
    bm = BrownianPath(t0=t0, w0=w0)
    for method in methods:
        _test_forward(sde, bm, method=method)
        _test_backward(sde, bm, method=method)


def _test_backward(sde, bm, method, adaptive=False, rtol=1e-3, atol=1e-2, eps=1e-7):
    # Must explicitly use `bm` to ensure determinism.

    def func(x):
        ys_and_logqp = sdeint_adjoint(sde, x, ts, bm, logqp=True, method=method, dt=dt, adaptive=adaptive)
        ys, logqp = ys_and_logqp
        # Just another arbitrarily chosen function with two outputs.
        return torch.stack([(ys ** 2.).sum(), (logqp / 3.).sum()], dim=0)

    # Finite-differences test.
    y0_ = y0.clone().requires_grad_(True)
    torch.autograd.gradcheck(func, y0_, rtol=rtol, atol=atol, eps=eps)


def _test_forward(sde, bm, method, adaptive=False, rtol=1e-6, atol=1e-5):
    sde.zero_grad()
    ys, log_ratio = sdeint_adjoint(
        sde, y0, ts, bm, logqp=True, method=method, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol)
    loss = ys.sum(0).mean(0).sum(0) + log_ratio.sum(0).mean(0)
    loss.backward()


if __name__ == '__main__':
    unittest.main()

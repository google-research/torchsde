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

"""Compare gradients computed with adjoint vs analytical solution."""
import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import unittest

import torch

from tests.basic_sde import BasicSDE1, BasicSDE2, BasicSDE3, BasicSDE4
from tests.problems import Ex1, Ex2, Ex3, Ex3Additive
from tests.torch_test import TorchTestCase
from torchsde import BrownianPath, sdeint_adjoint

torch.manual_seed(2147483647)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d = 10
m = 3
batch_size = 128
t0, t1 = ts = torch.tensor([0.0, 0.5]).to(device)
dt = 1e-3
y0 = torch.zeros(batch_size, d).to(device).fill_(0.1)
w0 = torch.zeros(batch_size, d).to(device)
methods = ('euler', 'milstein', 'srk')
adaptive_choices = (False, True)


class TestAdjoint(TorchTestCase):

    def test_ex1(self):
        problem = Ex1(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=False)

    def test_ex1_adaptive(self):
        problem = Ex1(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=True)

    def test_ex2(self):
        problem = Ex2(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=False)

    def test_ex2_adaptive(self):
        problem = Ex2(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=True)

    def test_ex3(self):
        problem = Ex3(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=False)

    def test_ex3_adaptive(self):
        problem = Ex3(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=True)

    def test_ex3_additive(self):
        problem = Ex3Additive(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=False)

    def test_ex3_additive_adaptive(self):
        problem = Ex3Additive(d).to(device)
        for method in methods:
            self._test_gradient(problem, method=method, adaptive=True)

    def test_basic_sde1(self):
        problem = BasicSDE1(d).to(device)
        for method in methods:
            for adaptive in adaptive_choices:
                self._test_basic(problem, method=method, adaptive=adaptive)

    def test_basic_sde2(self):
        problem = BasicSDE2(d).to(device)
        for method in methods:
            for adaptive in adaptive_choices:
                self._test_basic(problem, method=method, adaptive=adaptive)

    def test_basic_sde3(self):
        problem = BasicSDE3(d).to(device)
        for method in methods:
            for adaptive in adaptive_choices:
                self._test_basic(problem, method=method, adaptive=adaptive)

    def test_basic_sde4(self):
        problem = BasicSDE4(d).to(device)
        for method in methods:
            for adaptive in adaptive_choices:
                self._test_basic(problem, method, adaptive=adaptive)

    def _test_gradient(self, problem, method, adaptive, rtol=1e-5, atol=1e-4):
        if method == 'euler' and adaptive:
            return

        bm = BrownianPath(t0=t0, w0=w0)
        with torch.no_grad():
            grad_outputs = torch.ones(batch_size, d).to(device)
            alt_grad = problem.analytical_grad(y0, t1, grad_outputs, bm)

        problem.zero_grad()
        _, yt = sdeint_adjoint(problem, y0, ts, bm=bm, method=method, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol)
        loss = yt.sum(dim=1).mean(dim=0)
        loss.backward()
        adj_grad = torch.cat(tuple(p.grad for p in problem.parameters()))
        self.tensorAssertAllClose(alt_grad, adj_grad)

    def _test_basic(self, problem, method, adaptive, rtol=1e-5, atol=1e-4):
        if method == 'euler' and adaptive:
            return

        nbefore = _count_differentiable_params(problem)

        problem.zero_grad()
        _, yt = sdeint_adjoint(
            problem, y0, ts, method=method, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol
        )
        loss = yt.sum(dim=1).mean(dim=0)
        loss.backward()

        nafter = _count_differentiable_params(problem)
        self.assertEqual(nbefore, nafter)


def _count_differentiable_params(module):
    cnt = 0
    for p in module.parameters():
        if p.requires_grad:
            cnt += 1
    return cnt


if __name__ == '__main__':
    unittest.main()

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

import itertools
import unittest

import pytest
import torch

import torchsde
from .basic_sde import BasicSDE1, BasicSDE2, BasicSDE3, BasicSDE4
from .problems import Ex1, Ex2, Ex3
from .utils import assert_allclose

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.get_default_dtype()

ito_methods = {'milstein': 'ito', 'srk': 'ito'}
stratonovich_methods = {'midpoint': 'stratonovich', 'log_ode': 'stratonovich'}


@pytest.mark.parametrize("problem", [Ex1, Ex2, Ex3])
@pytest.mark.parametrize("method, sde_type", itertools.chain(ito_methods.items(), stratonovich_methods.items()))
@pytest.mark.parametrize("noise_type", ['diagonal', 'scalar', 'additive', 'general'])
@pytest.mark.parametrize('adaptive', (False, True))
def test_adjoint(problem, method, sde_type, noise_type, adaptive):
    if problem is not Ex3 and noise_type == 'additive':
        return
    if sde_type == 'ito' and noise_type == 'general':
        return
    if noise_type == "diagonal" and method == "log_ode":
        return

    d = 10
    m = {"scalar": 1}.get(noise_type, d)  # TODO: Decouple d from m.
    batch_size = 128
    t0, t1 = ts = torch.tensor([0.0, 0.5], device=device)
    y0 = torch.zeros(batch_size, d).to(device).fill_(0.1)
    v = torch.randn_like(y0)
    v /= v.norm(keepdim=True)  # Control the scale, so don't explode as batch size increases.
    problem = problem(d, sde_type=sde_type, noise_type=noise_type).to(device)
    levy_area_approximation = {
        'euler': 'none',
        'milstein': 'none',
        'srk': 'space-time',
        'midpoint': 'none',
        'log_ode': 'foster'
    }[method]
    bm = torchsde.BrownianInterval(
        t0=t0, t1=t1, shape=(batch_size, m), dtype=dtype, device=device,
        levy_area_approximation=levy_area_approximation
    )

    if hasattr(problem, "analytical_grad"):
        grad_true = problem.analytical_grad(y0=y0, t=t1, grad_output=v, bm=bm)
    else:
        # These gradients typically aren't accurate when adaptive==True.
        problem.zero_grad()
        _, y1 = torchsde.sdeint(problem, y0, ts, bm=bm, method=method, adaptive=adaptive)
        y1.backward(v)
        grad_true = torch.cat([p.grad.reshape(-1) for p in problem.parameters() if p.grad is not None])
        del y1

    problem.zero_grad()
    _, y1 = torchsde.sdeint_adjoint(problem, y0, ts, bm=bm, method=method, adaptive=adaptive)
    y1.backward(v)
    grad_adjoint = torch.cat([p.grad.reshape(-1) for p in problem.parameters() if p.grad is not None])

    assert_allclose(grad_true, grad_adjoint)


@pytest.mark.parametrize("problem", [BasicSDE1, BasicSDE2, BasicSDE3, BasicSDE4])
@pytest.mark.parametrize("method", ito_methods.keys())
@pytest.mark.parametrize('adaptive', (False, True))
def test_basic(problem, method, adaptive):
    if method == 'euler' and adaptive:
        return

    d = 10
    batch_size = 128
    ts = torch.tensor([0.0, 0.5], device=device)
    dt = 1e-3
    y0 = torch.zeros(batch_size, d).to(device).fill_(0.1)

    problem = problem(d).to(device)

    num_before = _count_differentiable_params(problem)

    problem.zero_grad()
    _, yt = torchsde.sdeint_adjoint(problem, y0, ts, method=method, dt=dt, adaptive=adaptive)
    loss = yt.sum(dim=1).mean(dim=0)
    loss.backward()

    num_after = _count_differentiable_params(problem)
    assert num_before == num_after


def _count_differentiable_params(module):
    return len([p for p in module.parameters() if p.requires_grad])


if __name__ == '__main__':
    unittest.main()

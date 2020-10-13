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
from . import utils

import torchsde
from torchsde.settings import NOISE_TYPES, METHODS
from .basic_sde import BasicSDE1, BasicSDE2, BasicSDE3, BasicSDE4
from .problems import Ex1, Ex2, Ex3, Ex4

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.get_default_dtype()

ito_methods = {'milstein': 'ito', 'srk': 'ito'}
stratonovich_methods = {'midpoint': 'stratonovich', 'log_ode': 'stratonovich'}


@pytest.mark.parametrize("sde_cls", [Ex1, Ex2, Ex3, Ex4])
@pytest.mark.parametrize("method, sde_type", itertools.chain(ito_methods.items(), stratonovich_methods.items()))
@pytest.mark.parametrize('adaptive', (False,))
def test_adjoint(sde_cls, method, sde_type, adaptive):
    # Skipping below, since method not supported for corresponding noise types.
    if method == METHODS.log_ode_midpoint and sde_cls.noise_type == NOISE_TYPES.diagonal:
        return
    if method == METHODS.milstein and sde_cls.noise_type == NOISE_TYPES.general:
        return
    if method == METHODS.srk and sde_cls.noise_type == NOISE_TYPES.general:
        return

    d = 5
    m = {
        NOISE_TYPES.scalar: 1,
        NOISE_TYPES.diagonal: d,
        NOISE_TYPES.general: 3,
        NOISE_TYPES.additive: 3
    }[sde_cls.noise_type]
    batch_size = 4
    t0, t1 = ts = torch.tensor([0.0, 0.5], device=device)
    dt = 1e-3
    y0 = torch.zeros(batch_size, d).to(device).fill_(0.1)
    sde = sde_cls(d=d, m=m, sde_type=sde_type).to(device)

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

    def func(inputs, modules):
        y0, sde = inputs[0], modules[0]
        ys = torchsde.sdeint_adjoint(sde, y0, ts, bm, dt=dt, method=method, adaptive=adaptive)
        return (ys[-1] ** 2).sum(dim=1).mean(dim=0)

    # `grad_inputs=True` also works, but we really only care about grad wrt params and want fast tests.
    utils.gradcheck(func, y0, sde, eps=1e-6, rtol=1e-2, atol=1e-3, grad_params=True)


@pytest.mark.parametrize("problem", [BasicSDE1, BasicSDE2, BasicSDE3, BasicSDE4])
@pytest.mark.parametrize("method", ito_methods.keys())
@pytest.mark.parametrize('adaptive', (False, True))
def test_basic(problem, method, adaptive):
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

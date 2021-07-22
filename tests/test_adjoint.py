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

import pytest
import torch
import torchsde
from torchsde.settings import LEVY_AREA_APPROXIMATIONS, METHODS, NOISE_TYPES, SDE_TYPES

from . import utils
from . import problems

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.get_default_dtype()


def _methods():
    yield SDE_TYPES.ito, METHODS.milstein, None
    yield SDE_TYPES.ito, METHODS.srk, None
    yield SDE_TYPES.stratonovich, METHODS.midpoint, None
    yield SDE_TYPES.stratonovich, METHODS.reversible_heun, None


@pytest.mark.parametrize("sde_cls", [problems.ExDiagonal, problems.ExScalar, problems.ExAdditive,
                                     problems.NeuralGeneral])
@pytest.mark.parametrize("sde_type, method, options", _methods())
@pytest.mark.parametrize('adaptive', (False,))
def test_against_numerical(sde_cls, sde_type, method, options, adaptive):
    # Skipping below, since method not supported for corresponding noise types.
    if sde_cls.noise_type == NOISE_TYPES.general and method in (METHODS.milstein, METHODS.srk):
        return

    d = 3
    m = {
        NOISE_TYPES.scalar: 1,
        NOISE_TYPES.diagonal: d,
        NOISE_TYPES.general: 2,
        NOISE_TYPES.additive: 2
    }[sde_cls.noise_type]
    batch_size = 4
    t0, t1 = ts = torch.tensor([0.0, 0.5], device=device)
    dt = 1e-3
    y0 = torch.full((batch_size, d), 0.1, device=device)
    sde = sde_cls(d=d, m=m, sde_type=sde_type).to(device)

    if method == METHODS.srk:
        levy_area_approximation = LEVY_AREA_APPROXIMATIONS.space_time
    else:
        levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none
    bm = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size, m), dtype=dtype, device=device,
        levy_area_approximation=levy_area_approximation
    )

    if method == METHODS.reversible_heun:
        tol = 1e-6
        adjoint_method = METHODS.adjoint_reversible_heun
        adjoint_options = options
    else:
        tol = 1e-2
        adjoint_method = None
        adjoint_options = None

    def func(inputs, modules):
        y0, sde = inputs[0], modules[0]
        ys = torchsde.sdeint_adjoint(sde, y0, ts, dt=dt, method=method, adjoint_method=adjoint_method,
                                     adaptive=adaptive, bm=bm, options=options, adjoint_options=adjoint_options)
        return (ys[-1] ** 2).sum(dim=1).mean(dim=0)

    # `grad_inputs=True` also works, but we really only care about grad wrt params and want fast tests.
    utils.gradcheck(func, y0, sde, eps=1e-6, rtol=tol, atol=tol, grad_params=True)


def _methods_dt_tol():
    for sde_type, method, options in _methods():
        if method == METHODS.reversible_heun:
            yield sde_type, method, options, 2**-3, 1e-3, 1e-4
            yield sde_type, method, options, 1e-3, 1e-3, 1e-4
        else:
            yield sde_type, method, options, 1e-3, 1e-2, 1e-2


@pytest.mark.parametrize("sde_cls", [problems.NeuralDiagonal, problems.NeuralScalar, problems.NeuralAdditive,
                                     problems.NeuralGeneral])
@pytest.mark.parametrize("sde_type, method, options, dt, rtol, atol", _methods_dt_tol())
@pytest.mark.parametrize("len_ts", [2, 9])
def test_against_sdeint(sde_cls, sde_type, method, options, dt, rtol, atol, len_ts):
    # Skipping below, since method not supported for corresponding noise types.
    if sde_cls.noise_type == NOISE_TYPES.general and method in (METHODS.milstein, METHODS.srk):
        return

    d = 3
    m = {
        NOISE_TYPES.scalar: 1,
        NOISE_TYPES.diagonal: d,
        NOISE_TYPES.general: 2,
        NOISE_TYPES.additive: 2
    }[sde_cls.noise_type]
    batch_size = 4
    ts = torch.linspace(0.0, 1.0, len_ts, device=device, dtype=torch.float64)
    t0 = ts[0]
    t1 = ts[-1]
    y0 = torch.full((batch_size, d), 0.1, device=device, dtype=torch.float64, requires_grad=True)
    sde = sde_cls(d=d, m=m, sde_type=sde_type).to(device, torch.float64)

    if method == METHODS.srk:
        levy_area_approximation = LEVY_AREA_APPROXIMATIONS.space_time
    else:
        levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none
    bm = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size, m), dtype=torch.float64, device=device,
        levy_area_approximation=levy_area_approximation
    )

    if method == METHODS.reversible_heun:
        adjoint_method = METHODS.adjoint_reversible_heun
        adjoint_options = options
    else:
        adjoint_method = None
        adjoint_options = None

    ys_true = torchsde.sdeint(sde, y0, ts, dt=dt, method=method, bm=bm, options=options)
    grad = torch.randn_like(ys_true)
    ys_true.backward(grad)

    true_grad = torch.cat([y0.grad.view(-1)] + [param.grad.view(-1) for param in sde.parameters()])
    y0.grad.zero_()
    for param in sde.parameters():
        param.grad.zero_()

    ys_test = torchsde.sdeint_adjoint(sde, y0, ts, dt=dt, method=method, bm=bm, adjoint_method=adjoint_method,
                                      options=options, adjoint_options=adjoint_options)
    ys_test.backward(grad)
    test_grad = torch.cat([y0.grad.view(-1)] + [param.grad.view(-1) for param in sde.parameters()])

    torch.testing.assert_allclose(ys_true, ys_test)
    torch.testing.assert_allclose(true_grad, test_grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("problem", [problems.BasicSDE1, problems.BasicSDE2, problems.BasicSDE3, problems.BasicSDE4])
@pytest.mark.parametrize("method", ["milstein", "srk"])
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

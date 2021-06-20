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

import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import pytest
import torch
import torchsde
from torchsde.settings import NOISE_TYPES

from . import problems

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

batch_size = 4
d = 3
m = 2
t0 = 0.0
t1 = 0.3
T = 5
dt = 0.05
dtype = torch.get_default_dtype()


class _nullcontext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.parametrize('device', devices)
def test_rename_methods(device):
    """Test renaming works with a subset of names."""
    sde = problems.CustomNamesSDE().to(device)
    y0 = torch.ones(batch_size, d, device=device)
    ts = torch.linspace(t0, t1, steps=T, device=device)
    ans = torchsde.sdeint(sde, y0, ts, dt=dt, names={'drift': 'forward'})
    assert ans.shape == (T, batch_size, d)


@pytest.mark.parametrize('device', devices)
def test_rename_methods_logqp(device):
    """Test renaming works with a subset of names when `logqp=True`."""
    sde = problems.CustomNamesSDELogqp().to(device)
    y0 = torch.ones(batch_size, d, device=device)
    ts = torch.linspace(t0, t1, steps=T, device=device)
    ans = torchsde.sdeint(sde, y0, ts, dt=dt, names={'drift': 'forward', 'prior_drift': 'w'}, logqp=True)
    assert ans[0].shape == (T, batch_size, d)
    assert ans[1].shape == (T - 1, batch_size)


def _use_bm__levy_area_approximation():
    yield False, None
    yield True, 'none'
    yield True, 'space-time'
    yield True, 'davie'
    yield True, 'foster'


@pytest.mark.parametrize('sde_type,method', [('ito', 'euler'), ('stratonovich', 'midpoint')])
def test_specialised_functions(sde_type, method):
    vector = torch.randn(m)
    fg = problems.FGSDE(sde_type, vector)
    f_and_g = problems.FAndGSDE(sde_type, vector)
    g_prod = problems.GProdSDE(sde_type, vector)
    f_and_g_prod = problems.FAndGProdSDE(sde_type, vector)
    f_and_g_with_g_prod1 = problems.FAndGGProdSDE1(sde_type, vector)
    f_and_g_with_g_prod2 = problems.FAndGGProdSDE2(sde_type, vector)

    y0 = torch.randn(batch_size, d)

    outs = []
    for sde in (fg, f_and_g, g_prod, f_and_g_prod, f_and_g_with_g_prod1, f_and_g_with_g_prod2):
        bm = torchsde.BrownianInterval(t0, t1, (batch_size, m), entropy=45678)
        outs.append(torchsde.sdeint(sde, y0, [t0, t1], dt=dt, bm=bm)[1])
    for o in outs[1:]:
        # Equality of floating points, because we expect them to do everything exactly the same.
        assert o.shape == outs[0].shape
        assert (o == outs[0]).all()


@pytest.mark.parametrize('sde_cls', [problems.ExDiagonal, problems.ExScalar, problems.ExAdditive,
                                     problems.NeuralGeneral])
@pytest.mark.parametrize('use_bm,levy_area_approximation', _use_bm__levy_area_approximation())
@pytest.mark.parametrize('sde_type', ['ito', 'stratonovich'])
@pytest.mark.parametrize('method',
                         ['blah', 'euler', 'milstein', 'milstein_grad_free', 'srk', 'euler_heun', 'heun', 'midpoint',
                          'log_ode'])
@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('logqp', [True, False])
@pytest.mark.parametrize('device', devices)
def test_sdeint_run_shape_method(sde_cls, use_bm, levy_area_approximation, sde_type, method, adaptive, logqp, device):
    """Tests that sdeint:
    (a) runs/raises an error as appropriate
    (b) produces tensors of the right shape
    (c) accepts every method
    """

    if method == 'milstein_grad_free':
        method = 'milstein'
        options = dict(grad_free=True)
    else:
        options = dict()

    should_fail = False
    if sde_type == 'ito':
        if method not in ('euler', 'srk', 'milstein'):
            should_fail = True
    else:
        if method not in ('euler_heun', 'heun', 'midpoint', 'log_ode', 'milstein'):
            should_fail = True
    if method in ('milstein', 'srk') and sde_cls.noise_type == 'general':
        should_fail = True
    if method == 'srk' and levy_area_approximation == 'none':
        should_fail = True
    if method == 'log_ode' and levy_area_approximation in ('none', 'space-time'):
        should_fail = True

    if sde_cls.noise_type in (NOISE_TYPES.scalar, NOISE_TYPES.diagonal):
        kwargs = {'d': d}
    else:
        kwargs = {'d': d, 'm': m}
    sde = sde_cls(sde_type=sde_type, **kwargs).to(device)

    if use_bm:
        if sde_cls.noise_type == 'scalar':
            size = (batch_size, 1)
        elif sde_cls.noise_type == 'diagonal':
            size = (batch_size, d + 1) if logqp else (batch_size, d)
        else:
            assert sde_cls.noise_type in ('additive', 'general')
            size = (batch_size, m)
        bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=size, dtype=dtype, device=device,
                                       levy_area_approximation=levy_area_approximation)
    else:
        bm = None

    _test_sdeint(sde, bm, method, adaptive, logqp, device, should_fail, options)


@pytest.mark.parametrize("sde_cls", [problems.BasicSDE1, problems.BasicSDE2, problems.BasicSDE3, problems.BasicSDE4])
@pytest.mark.parametrize('method', ['euler', 'milstein', 'milstein_grad_free', 'srk'])
@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('device', devices)
def test_sdeint_dependencies(sde_cls, method, adaptive, device):
    """This test uses diagonal noise. This checks if the solvers still work when some of the functions don't depend on
    the states/params and when some states/params don't require gradients.
    """

    if method == 'milstein_grad_free':
        method = 'milstein'
        options = dict(grad_free=True)
    else:
        options = dict()

    sde = sde_cls(d=d).to(device)
    bm = None
    logqp = False
    should_fail = False
    _test_sdeint(sde, bm, method, adaptive, logqp, device, should_fail, options)


def _test_sdeint(sde, bm, method, adaptive, logqp, device, should_fail, options):
    y0 = torch.ones(batch_size, d, device=device)
    ts = torch.linspace(t0, t1, steps=T, device=device)
    if adaptive and method == 'euler' and sde.noise_type != 'additive':
        ctx = pytest.warns(UserWarning)
    else:
        ctx = _nullcontext()

    # Using `f` as drift.
    with torch.no_grad():
        try:
            with ctx:
                ans = torchsde.sdeint(sde, y0, ts, bm, method=method, dt=dt, adaptive=adaptive, logqp=logqp,
                                      options=options)
        except ValueError:
            if should_fail:
                return
            raise
        else:
            if should_fail:
                pytest.fail("Expected an error; did not get one.")
    if logqp:
        ans, log_ratio = ans
        assert log_ratio.shape == (T - 1, batch_size)
    assert ans.shape == (T, batch_size, d)

    # Using `h` as drift.
    with torch.no_grad():
        with ctx:
            ans = torchsde.sdeint(sde, y0, ts, bm, method=method, dt=dt, adaptive=adaptive, names={'drift': 'h'},
                                  logqp=logqp, options=options)
    if logqp:
        ans, log_ratio = ans
        assert log_ratio.shape == (T - 1, batch_size)
    assert ans.shape == (T, batch_size, d)


@pytest.mark.parametrize("sde_cls", [problems.NeuralDiagonal, problems.NeuralScalar, problems.NeuralAdditive,
                                     problems.NeuralGeneral])
def test_reversibility(sde_cls):
    batch_size = 32
    state_size = 4
    t_size = 20
    dt = 0.1

    brownian_size = {
        NOISE_TYPES.scalar: 1,
        NOISE_TYPES.diagonal: state_size,
        NOISE_TYPES.general: 2,
        NOISE_TYPES.additive: 2
    }[sde_cls.noise_type]

    class MinusSDE(torch.nn.Module):
        def __init__(self, sde):
            self.noise_type = sde.noise_type
            self.sde_type = sde.sde_type
            self.f = lambda t, y: -sde.f(-t, y)
            self.g = lambda t, y: -sde.g(-t, y)

    sde = sde_cls(d=state_size, m=brownian_size, sde_type='stratonovich')
    minus_sde = MinusSDE(sde)
    y0 = torch.full((batch_size, state_size), 0.1)
    ts = torch.linspace(0, (t_size - 1) * dt, t_size)
    bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_size, brownian_size))
    ys, (f, g, z) = torchsde.sdeint(sde, y0, ts, bm=bm, method='reversible_heun', dt=dt, extra=True)
    backward_ts = -ts.flip(0)
    backward_ys = torchsde.sdeint(minus_sde, ys[-1], backward_ts, bm=torchsde.ReverseBrownian(bm),
                                  method='reversible_heun', dt=dt, extra_solver_state=(-f, -g, z))
    backward_ys = backward_ys.flip(0)

    torch.testing.assert_allclose(ys, backward_ys, rtol=1e-6, atol=1e-6)

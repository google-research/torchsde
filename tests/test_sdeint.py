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


@pytest.mark.parametrize('sde_cls', [problems.Ex1, problems.Ex2, problems.Ex3, problems.Neural4])
@pytest.mark.parametrize('use_bm,levy_area_approximation', _use_bm__levy_area_approximation())
@pytest.mark.parametrize('sde_type', ['ito', 'stratonovich'])
@pytest.mark.parametrize('method', ['blah', 'euler', 'milstein', 'srk', 'euler_heun', 'heun', 'midpoint', 'log_ode'])
@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('logqp', [True, False])
@pytest.mark.parametrize('device', devices)
def test_sdeint_run_shape_method(sde_cls, use_bm, levy_area_approximation, sde_type, method, adaptive, logqp, device):
    """Tests that sdeint:
    (a) runs/raises an error as appropriate
    (b) produces tensors of the right shape
    (c) accepts every method
    """
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

    _test_sdeint(sde, bm, method, adaptive, logqp, device, should_fail)


@pytest.mark.parametrize("sde_cls", [problems.BasicSDE1, problems.BasicSDE2, problems.BasicSDE3, problems.BasicSDE4])
@pytest.mark.parametrize('method', ['euler', 'milstein', 'srk'])
@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('device', devices)
def test_sdeint_dependencies(sde_cls, method, adaptive, device):
    """This test uses diagonal noise. This checks if the solvers still work when some of the functions don't depend on
    the states/params and when some states/params don't require gradients.
    """

    sde = sde_cls(d=d).to(device)
    bm = None
    logqp = False
    should_fail = False
    _test_sdeint(sde, bm, method, adaptive, logqp, device, should_fail)


def _test_sdeint(sde, bm, method, adaptive, logqp, device, should_fail):
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
                ans = torchsde.sdeint(sde, y0, ts, bm, method=method, dt=dt, adaptive=adaptive, logqp=logqp)
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
            ans = torchsde.sdeint(
                sde, y0, ts, bm, method=method, dt=dt, adaptive=adaptive, names={'drift': 'h'}, logqp=logqp)
    if logqp:
        ans, log_ratio = ans
        assert log_ratio.shape == (T - 1, batch_size)
    assert ans.shape == (T, batch_size, d)

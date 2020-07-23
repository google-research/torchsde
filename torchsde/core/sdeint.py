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

import warnings

import torch

try:
    from torchsde.brownian_lib import BrownianPath
except Exception:
    from torchsde.brownian.brownian_path import BrownianPath

from torchsde.core import base_sde
from torchsde.core import methods
from torchsde.core import settings


def sdeint(sde, y0, ts, bm=None, logqp=False, method='srk', dt=1e-3, adaptive=False, rtol=1e-6, atol=1e-5, dt_min=1e-4,
           options=None, names=None):
    """Numerically integrate an Itô SDE.

    Args:
        sde: An object with the methods `f` and `g` representing the drift and diffusion functions. The methods
            should take in time `t` and state `y` and return a tensor or tuple of tensors. The output signature of
            `f` should match `y`. The output of `g` should either be a single (or a tuple) of tensors of size
            (batch_size, d) for diagonal noise problems or (batch_size, d, m) for other problem types,
            where d is the dimensionality of state and m is the dimensionality of the Brownian motion.
        y0: A single (or a tuple) of tensors of size (batch_size, d).
        ts: A list or 1-D tensor in non-descending order.
        bm: A `BrownianPath` or `BrownianTree` object. Defaults to `BrownianPath` for diagonal noise residing on CPU.
        logqp: If True, also return the Radon-Nikodym derivative, which is a log-ratio penalty across the whole path.
        method: Numerical integration method, one of (`euler`, `milstein`, `srk`). Defaults to `srk`.
        dt: A float for the constant step size or initial step size for adaptive time-stepping.
        adaptive: If True, use adaptive time-stepping.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        dt_min: Minimum step size for adaptive time-stepping.
        options: Optional dict of configuring options for the indicated integration method.
        names: Optional dict of method names to use as drift, diffusion, and prior drift. Expected keys are `drift`,
            `diffusion`, `prior_drift`.

    Returns:
        A single state tensor of size (T, batch_size, d) or a tuple of such tensors. Also returns a single log-ratio
        tensor of size (T - 1, batch_size) or a tuple of such tensors, if logqp=True.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method, or sde module missing required methods.
    """
    names_to_change = get_names_to_change(names)
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)
    check_contract(sde=sde, method=method, adaptive=adaptive, logqp=logqp)

    if bm is None:
        bm = BrownianPath(t0=ts[0], w0=torch.zeros_like(y0).cpu())

    tensor_input = isinstance(y0, torch.Tensor)
    if tensor_input:
        sde = base_sde.TupleSDE(sde)
        y0 = (y0,)
        bm_ = bm
        bm = lambda t: (bm_(t),)

    sde = base_sde.ForwardSDEIto(sde)
    results = integrate(
        sde=sde, y0=y0, ts=ts, bm=bm, method=method, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol, dt_min=dt_min,
        options=options, logqp=logqp
    )
    if not logqp and tensor_input:
        return results[0]
    return results


def get_names_to_change(names):
    if names is None:
        return {}

    keys = ('drift', 'diffusion', 'prior_drift')
    return {key: names[key] for key in keys if key in names}


def check_contract(sde, method, adaptive, logqp, adjoint_method=None):
    required_funcs = ('f', 'g', 'h') if logqp else ('f', 'g')
    missing_funcs = tuple(func for func in required_funcs if not hasattr(sde, func))
    if len(missing_funcs) > 0:
        raise ValueError(f'sde is required to have the methods {required_funcs}. Missing functions: {missing_funcs}')

    if not hasattr(sde, 'noise_type'):
        raise ValueError(f'sde does not have the attribute noise_type.')

    if sde.noise_type not in settings.NOISE_TYPES:
        raise ValueError(f'Expected noise type in {settings.NOISE_TYPES}, but found {sde.noise_type}.')

    if not hasattr(sde, 'sde_type'):
        raise ValueError(f'sde does not have the attribute sde_type.')

    if sde.sde_type not in settings.SDE_TYPES:
        raise ValueError(f'Expected sde type in {settings.SDE_TYPES}, but found {sde.sde_type}.')

    if method not in settings.METHODS:
        raise ValueError(f'Expected method in {settings.METHODS}, but found {method}.')

    if adjoint_method is not None:
        if adjoint_method not in settings.METHODS:
            raise ValueError(f'Expected adjoint_method in {settings.METHODS}, but found {method}.')

    # TODO: This warning should be based on the `strong_order` attribute of the solver.
    if adaptive and method == 'euler' and sde.noise_type != "additive":
        warnings.warn(f'Numerical solution is only guaranteed to converge to the correct solution '
                      f'when a strong order >=1.0 scheme is used for adaptive time-stepping.')


def integrate(sde, y0, ts, bm, method, dt, adaptive, rtol, atol, dt_min, options, logqp=False):
    if options is None:
        options = {}

    solver_fn = _select(method=method, noise_type=sde.noise_type)
    solver = solver_fn(
        sde=sde, bm=bm, y0=y0, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol, dt_min=dt_min, options=options)
    if logqp:
        return solver.integrate_logqp(ts)
    return solver.integrate(ts)


def _select(method, noise_type):
    if noise_type == 'diagonal':
        return {
            'euler': methods.EulerDiagonal,
            'milstein': methods.MilsteinDiagonal,
            'srk': methods.SRKDiagonal
        }[method]
    elif noise_type == "general":
        if method != 'euler':
            raise ValueError('For SDEs with general noise only the Euler method is supported.')
        return {
            'euler': methods.EulerGeneral,
        }[method]
    elif noise_type == "additive":
        return {
            'euler': methods.EulerAdditive,
            'milstein': methods.EulerAdditive,  # Milstein same as Euler since diffusion is constant.
            'srk': methods.SRKAdditive,
        }[method]
    elif noise_type == "scalar":
        return {
            'euler': methods.EulerScalar,
            'milstein': methods.MilsteinScalar,
            'srk': methods.SRKScalar,
        }[method]
    else:
        exit(1)

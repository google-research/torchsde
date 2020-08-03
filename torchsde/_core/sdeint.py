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
from typing import Optional, Dict, Any

import torch

try:
    from torchsde.brownian_lib import BrownianPath
except Exception:  # noqa
    from torchsde._brownian.brownian_path import BrownianPath  # noqa

from torchsde._brownian.base_brownian import Brownian  # noqa
from . import base_sde
from . import methods
from . import settings
from .types import TensorOrTensors, Scalar, Vector


def sdeint(sde,
           y0: TensorOrTensors,
           ts: Vector,
           bm: Optional[Brownian] = None,
           logqp: Optional[bool] = False,
           method: Optional[str] = 'srk',
           dt: Optional[Scalar] = 1e-3,
           adaptive: Optional[bool] = False,
           rtol: Optional[float] = 1e-5,
           atol: Optional[float] = 1e-4,
           dt_min: Optional[Scalar] = 1e-5,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None):
    """Numerically integrate an Itô SDE.

    Args:
        sde (object): Object with methods `f` and `g` representing the drift and
            diffusion. The output of `g` should be a single (or a tuple of)
            tensor(s) of size (batch_size, d) for diagonal noise SDEs or
            (batch_size, d, m) for SDEs of other noise types; d is the
            dimensionality of state and m is the dimensionality of Brownian
            motion.
        y0 (sequence of Tensor): Tensors for initial state.
        ts (Tensor or sequence of float): Query times in non-descending order.
            The state at the first time of `ts` should be `y0`.
        bm (Brownian, optional): A `BrownianPath` or `BrownianTree` object.
            Should return tensors of size (batch_size, m) for `__call__`.
            Defaults to `BrownianPath` for diagonal noise on CPU.
            Currently does not support tuple outputs yet.
        logqp (bool, optional): If `True`, also return the log-ratio penalty.
        method (str, optional): Name of numerical integration method.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        rtol (float, optional): Relative tolerance.
        atol (float, optional): Absolute tolerance.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        names (dict, optional): Dict of method names for drift, diffusion, and
            prior drift. Expected keys are "drift", "diffusion", and
            "prior_drift". Serves so that users can use methods with names not
            in `("f", "g", "h")`, e.g. to use the method "foo" for the drift,
            we would supply `names={"drift": "foo"}`.

    Returns:
        A single state tensor of size (T, batch_size, d) or a tuple of such
        tensors. Also returns a single log-ratio tensor of size
        (T - 1, batch_size) or a tuple of such tensors, if `logqp==True`.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or `sde` is missing required methods.
    """
    names_to_change = get_names_to_change(names)
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)
    check_contract(sde=sde, method=method, logqp=logqp)

    if bm is None:
        bm = BrownianPath(t0=ts[0], w0=torch.zeros_like(y0).cpu())

    tensor_input = isinstance(y0, torch.Tensor)
    if tensor_input:
        sde = base_sde.TupleSDE(sde)
        y0 = (y0,)
        bm_ = bm
        bm = lambda t: (bm_(t),)  # noqa

    sde = base_sde.ForwardSDEIto(sde)
    results = integrate(
        sde=sde,
        y0=y0,
        ts=ts,
        bm=bm,
        method=method,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options,
        logqp=logqp
    )
    if not logqp and tensor_input:
        return results[0]
    return results


def get_names_to_change(names):
    if names is None:
        return {}

    keys = ('drift', 'diffusion', 'prior_drift')
    return {key: names[key] for key in keys if key in names}


def check_contract(sde, method, logqp, adjoint_method=None):
    required_funcs = ('f', 'g', 'h') if logqp else ('f', 'g')
    missing_funcs = [func for func in required_funcs if not hasattr(sde, func)]
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


def integrate(sde, y0, ts, bm, method, dt, adaptive, rtol, atol, dt_min, options, logqp=False):
    if options is None:
        options = {}

    solver_fn = _select(method=method, noise_type=sde.noise_type)
    solver = solver_fn(
        sde=sde,
        bm=bm,
        y0=y0,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options
    )
    if adaptive and solver.strong_order < 1.0:
        warnings.warn(f'Numerical solution is only guaranteed to converge to the correct solution '
                      f'when a strong order >=1.0 scheme is used for adaptive time-stepping.')
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
            'milstein': methods.EulerAdditive,
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

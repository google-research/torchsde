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

import warnings
from typing import Optional, Dict, Any

import torch

from . import base_sde
from . import methods
from .._brownian import BaseBrownian, TupleBrownian, BrownianInterval
from ..settings import SDE_TYPES, NOISE_TYPES, METHODS, LEVY_AREA_APPROXIMATIONS
from ..types import TensorOrTensors, Scalar, Vector


def sdeint(sde: [base_sde.BaseSDE],
           y0: TensorOrTensors,
           ts: Vector,
           bm: Optional[BaseBrownian] = None,
           logqp: Optional[bool] = False,
           method: Optional[str] = 'srk',
           dt: Optional[Scalar] = 1e-3,
           adaptive: Optional[bool] = False,
           rtol: Optional[float] = 1e-5,
           atol: Optional[float] = 1e-4,
           dt_min: Optional[Scalar] = 1e-5,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None) -> TensorOrTensors:
    """Numerically integrate an Itô SDE.

    Args:
        sde: Object with methods `f` and `g` representing the drift and
            diffusion. The output of `g` should be a single (or a tuple of)
            tensor(s) of size (batch_size, d) for diagonal noise SDEs or
            (batch_size, d, m) for SDEs of other noise types; d is the
            dimensionality of state and m is the dimensionality of Brownian
            motion.
        y0 (sequence of Tensor): Tensors for initial state.
        ts (Tensor or sequence of float): Query times in non-descending order.
            The state at the first time of `ts` should be `y0`.
        bm (Brownian, optional): A 'BrownianInterval', `BrownianPath` or
            `BrownianTree` object. Should return tensors of size (batch_size, m)
            for `__call__`. Defaults to `BrownianInterval`. Currently does not
            support tuple outputs yet.
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
            or if `sde` is missing required methods.
    """
    sde, y0, bm, tensor_input = check_contract(sde=sde, method=method, logqp=logqp, ts=ts, y0=y0, bm=bm, names=names)

    sde = base_sde.ForwardSDE(sde)
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


def check_contract(sde, method, logqp, ts, y0, bm, names):
    if names is None:
        names_to_change = {}
    else:
        names_to_change = {key: names[key] for key in ('drift', 'diffusion', 'prior_drift') if key in names}
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)

    required_funcs = ('f', 'g', 'h') if logqp else ('f', 'g')
    missing_funcs = [func for func in required_funcs if not hasattr(sde, func)]
    if len(missing_funcs) > 0:
        raise ValueError(f'sde is required to have the methods {required_funcs}. Missing functions: {missing_funcs}')

    if not hasattr(sde, 'noise_type'):
        raise ValueError(f'sde does not have the attribute noise_type.')

    if sde.noise_type not in NOISE_TYPES:
        raise ValueError(f'Expected noise type in {NOISE_TYPES}, but found {sde.noise_type}.')

    if not hasattr(sde, 'sde_type'):
        raise ValueError(f'sde does not have the attribute sde_type.')

    if sde.sde_type not in SDE_TYPES:
        raise ValueError(f'Expected sde type in {SDE_TYPES}, but found {sde.sde_type}.')

    if method not in METHODS:
        raise ValueError(f'Expected method in {METHODS}, but found {method}.')

    tensor_input = torch.is_tensor(y0)
    if tensor_input:
        sde = base_sde.TupleSDE(sde)
        y0 = (y0,)
    if not isinstance(y0, tuple) or any(not torch.is_tensor(y0_) for y0_ in y0):
        raise ValueError("y0 must be a Tensor or a tuple of Tensors.")

    drift_shape = [fi.shape for fi in sde.f(ts[0], y0)]
    diffusion_shape = [gi.shape for gi in sde.g(ts[0], y0)]

    if len(drift_shape) != len(diffusion_shape) or len(drift_shape) != len(y0):
        raise ValueError("drift, diffusion and y0 must all return the same number of Tensors.")

    for drift_shape_, y0_ in zip(drift_shape, y0):
        if drift_shape_ != y0_.shape:
            raise ValueError(f"Drift must return a Tensor of the same shape as y0. Got drift shape {drift_shape_} but "
                             f"y0 shape {y0_.shape}.")

    noise_channels = diffusion_shape[0][-1]
    if sde.noise_type in (NOISE_TYPES.additive, NOISE_TYPES.general, NOISE_TYPES.scalar):
        batch_dimensions = diffusion_shape[0][:-2]
        for drift_shape_, diffusion_shape_ in zip(drift_shape, diffusion_shape):
            drift_shape_ = tuple(drift_shape_)
            diffusion_shape_ = tuple(diffusion_shape_)
            if len(drift_shape_) == 0:
                raise ValueError("Drift must be of shape (..., state_channels), but got shape ().")
            if len(diffusion_shape_) < 2:
                raise ValueError(f"Diffusion must have shape (..., state_channels, noise_channels), but got shape "
                                 f"{diffusion_shape_}.")
            if drift_shape_ != diffusion_shape_[:-1]:
                raise ValueError(f"Drift and diffusion shapes do not match. Got drift shape "
                                 f"{drift_shape_}, meaning {drift_shape_[:-1]} batch dimensions and {drift_shape_[-1]} "
                                 f"channel dimensions, but diffusion shape {diffusion_shape_}, meaning "
                                 f"{diffusion_shape_[:-2]} batch dimensions, {diffusion_shape_[-2]} channel dimensions "
                                 f"and {diffusion_shape_[-1]} noise dimension.")
            if diffusion_shape_[:-2] != batch_dimensions:
                raise ValueError("Every Tensor return by the diffusion must have the same number and size of batch "
                                 "dimensions.")
            if diffusion_shape_[-1] != noise_channels:
                raise ValueError("Every Tensor return by the diffusion must have the same number of noise channels.")
        if sde.noise_type == NOISE_TYPES.scalar:
            if noise_channels != 1:
                raise ValueError(f"Scalar noise must have only one channel; the diffusion has {noise_channels} noise "
                                 f"channels.")
    else:  # sde.noise_type == NOISE_TYPES.diagonal
        batch_dimensions = diffusion_shape[0][:-1]
        for drift_shape_, diffusion_shape_ in zip(drift_shape, diffusion_shape):
            drift_shape_ = tuple(drift_shape_)
            diffusion_shape_ = tuple(diffusion_shape_)
            if len(drift_shape_) == 0:
                raise ValueError("Drift must be of shape (..., state_channels), but got shape ().")
            if len(diffusion_shape_) == 0:
                raise ValueError(f"Diffusion must have shape (..., state_channels), but got shape ().")
            if drift_shape_ != diffusion_shape_:
                raise ValueError(f"Drift and diffusion shapes do not match. Got drift shape "
                                 f"{drift_shape_}, meaning {drift_shape_[:-1]} batch dimensions and {drift_shape_[-1]} "
                                 f"channel dimensions, but diffusion shape {diffusion_shape_}, meaning "
                                 f"{diffusion_shape_[:-1]} batch dimensions, {diffusion_shape_[-1]} channel dimensions "
                                 f"and {diffusion_shape_[-1]} noise dimension.")
            if diffusion_shape_[:-1] != batch_dimensions:
                raise ValueError("Every Tensor return by the diffusion must have the same number and size of batch "
                                 "dimensions.")
            if diffusion_shape_[-1] != noise_channels:
                raise ValueError("Every Tensor return by the diffusion must have the same number of noise channels.")

    if bm is None:
        if method == METHODS.srk:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.space_time
        else:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], shape=(*batch_dimensions, noise_channels), dtype=y0[0].dtype,
                              device=y0[0].device, levy_area_approximation=levy_area_approximation)
    if tensor_input:
        bm = TupleBrownian(bm)

    return sde, y0, bm, tensor_input


def integrate(sde, y0, ts, bm, method, dt, adaptive, rtol, atol, dt_min, options, logqp=False):
    if options is None:
        options = {}

    solver_fn = methods.select(method=method)
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

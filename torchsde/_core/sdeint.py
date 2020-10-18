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

import torch

from . import base_sde
from . import methods
from . import misc
from .._brownian import BaseBrownian, BrownianInterval
from ..settings import LEVY_AREA_APPROXIMATIONS, METHODS, NOISE_TYPES, SDE_TYPES
from ..types import Any, Dict, Optional, Scalar, Tensor, TensorOrTensors, Vector


def sdeint(sde: base_sde.BaseSDE,
           y0: Tensor,
           ts: Vector,
           bm: Optional[BaseBrownian] = None,
           method: Optional[str] = None,
           dt: Optional[Scalar] = 1e-3,
           adaptive: Optional[bool] = False,
           rtol: Optional[Scalar] = 1e-5,
           atol: Optional[Scalar] = 1e-4,
           dt_min: Optional[Scalar] = 1e-5,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None,
           logqp: Optional[bool] = False,
           **unused_kwargs) -> TensorOrTensors:
    """Numerically integrate an SDE.

    Args:
        sde: Object with methods `f` and `g` representing the
            drift and diffusion. The output of `g` should be a single tensor of
            size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m)
            for SDEs of other noise types; d is the dimensionality of state and
            m is the dimensionality of Brownian motion.
        y0 (Tensor): A tensor for the initial state.
        ts (Tensor or sequence of float): Query times in non-descending order.
            The state at the first time of `ts` should be `y0`.
        bm (Brownian, optional): A 'BrownianInterval', `BrownianPath` or
            `BrownianTree` object. Should return tensors of size (batch_size, m)
            for `__call__`. Defaults to `BrownianInterval`.
        method (str, optional): Numerical integration method to use. Must be
            compatible with the SDE type (Ito/Stratonovich) and the noise type
            (scalar/additive/diagonal/general). Defaults to a sensible choice
            depending on the SDE type and noise type of the supplied SDE.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        rtol (float, optional): Relative tolerance.
        atol (float, optional): Absolute tolerance.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        names (dict, optional): Dict of method names for drift and diffusion.
            Expected keys are "drift" and "diffusion". Serves so that users can
            use methods with names not in `("f", "g")`, e.g. to use the
            method "foo" for the drift, we supply `names={"drift": "foo"}`.
        logqp (bool, optional): If `True`, also return the log-ratio penalty.
            This argument will be deprecated in the future and is only included
            to support backward compatibility.

    Returns:
        A single state tensor of size (T, batch_size, d).

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or if `sde` is missing required methods.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint`")
    del unused_kwargs

    sde, y0, ts, bm, method = check_contract(sde, y0, ts, bm, method, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'atol', 'dt_min'],
                        [ts, dt, rtol, atol, dt_min])
    return integrate(
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


def check_contract(sde, y0, ts, bm, method, names, logqp):
    if names is None:
        names_to_change = {}
    else:
        names_to_change = {key: names[key] for key in ("drift", "diffusion", "prior_drift") if key in names}
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)

    required_funcs = ("f", "g", "h") if logqp else ("f", "g")
    missing_funcs = [func for func in required_funcs if not hasattr(sde, func)]
    if len(missing_funcs) > 0:
        raise ValueError(f"sde is required to have the methods {required_funcs}. Missing functions: {missing_funcs}")

    # --- Backwards compatibility: v0.1.1. ---
    if logqp:
        sde = base_sde.SDELogqp(sde)
        y0 = torch.cat((y0, y0.new_zeros(size=(y0.size(0), 1))), dim=1)
    # ----------------------------------------

    if not hasattr(sde, "noise_type"):
        raise ValueError(f"sde does not have the attribute noise_type.")

    if sde.noise_type not in NOISE_TYPES:
        raise ValueError(f"Expected noise type in {NOISE_TYPES}, but found {sde.noise_type}.")

    if not hasattr(sde, "sde_type"):
        raise ValueError(f"sde does not have the attribute sde_type.")

    if sde.sde_type not in SDE_TYPES:
        raise ValueError(f"Expected sde type in {SDE_TYPES}, but found {sde.sde_type}.")

    if method is None:
        method = {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: METHODS.srk,
                NOISE_TYPES.additive: METHODS.srk,
                NOISE_TYPES.scalar: METHODS.srk,
                NOISE_TYPES.general: METHODS.euler
            }[sde.noise_type],
            SDE_TYPES.stratonovich: METHODS.midpoint,
        }[sde.sde_type]

    if method not in METHODS:
        raise ValueError(f"Expected method in {METHODS}, but found {method}.")

    if not torch.is_tensor(y0):
        raise ValueError(f"`y0` must be a torch.Tensor.")

    if not torch.is_tensor(ts):
        if not isinstance(ts, (tuple, list)) or not all(isinstance(t, (float, int)) for t in ts):
            raise ValueError(f"Evaluation times `ts` must be a 1-D Tensor or list/tuple of floats.")
        ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)

    drift_shape = sde.f(ts[0], y0).size()
    if drift_shape != y0.size():
        raise ValueError(f"Drift must return a Tensor of the same shape as `y0`. "
                         f"Got drift shape {drift_shape}, but y0 shape {y0.size()}.")

    diffusion_shape = sde.g(ts[0], y0).size()
    noise_channels = diffusion_shape[-1]
    if sde.noise_type in (NOISE_TYPES.additive, NOISE_TYPES.general, NOISE_TYPES.scalar):
        batch_dimensions = diffusion_shape[:-2]
        drift_shape, diffusion_shape = tuple(drift_shape), tuple(diffusion_shape)
        if len(drift_shape) == 0:
            raise ValueError("Drift must be of shape (..., state_channels), but got shape ().")
        if len(diffusion_shape) < 2:
            raise ValueError(f"Diffusion must have shape (..., state_channels, noise_channels), "
                             f"but got shape {diffusion_shape}.")
        if drift_shape != diffusion_shape[:-1]:
            raise ValueError(f"Drift and diffusion shapes do not match. Got drift shape {drift_shape}, "
                             f"meaning {drift_shape[:-1]} batch dimensions and {drift_shape[-1]} channel "
                             f"dimensions, but diffusion shape {diffusion_shape}, meaning "
                             f"{diffusion_shape[:-2]} batch dimensions, {diffusion_shape[-2]} channel "
                             f"dimensions and {diffusion_shape[-1]} noise dimension.")
        if diffusion_shape[:-2] != batch_dimensions:
            raise ValueError("Every Tensor returned by the diffusion must have the same number and size of batch "
                             "dimensions.")
        if diffusion_shape[-1] != noise_channels:
            raise ValueError("Every Tensor returned by the diffusion must have the same number of noise channels.")
        if sde.noise_type == NOISE_TYPES.scalar:
            if noise_channels != 1:
                raise ValueError(f"Scalar noise must have only one channel; "
                                 f"the diffusion has {noise_channels} noise channels.")
    else:  # sde.noise_type == NOISE_TYPES.diagonal
        batch_dimensions = diffusion_shape[:-1]
        drift_shape, diffusion_shape = tuple(drift_shape), tuple(diffusion_shape)
        if len(drift_shape) == 0:
            raise ValueError("Drift must be of shape (..., state_channels), but got shape ().")
        if len(diffusion_shape) == 0:
            raise ValueError(f"Diffusion must have shape (..., state_channels), but got shape ().")
        if drift_shape != diffusion_shape:
            raise ValueError(f"Drift and diffusion shapes do not match. Got drift shape {drift_shape}, "
                             f"meaning {drift_shape[:-1]} batch dimensions and {drift_shape[-1]} channel "
                             f"dimensions, but diffusion shape {diffusion_shape}, meaning "
                             f"{diffusion_shape[:-1]} batch dimensions, {diffusion_shape[-1]} channel "
                             f"dimensions and {diffusion_shape[-1]} noise dimension.")
        if diffusion_shape[:-1] != batch_dimensions:
            raise ValueError("Every Tensor return by the diffusion must have the same number and size of batch "
                             "dimensions.")
        if diffusion_shape[-1] != noise_channels:
            raise ValueError("Every Tensor return by the diffusion must have the same number of noise "
                             "channels.")
    sde = base_sde.ForwardSDE(sde)

    if bm is None:
        if method == METHODS.srk:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.space_time
        elif method == METHODS.log_ode_midpoint:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.foster
        else:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=(*batch_dimensions, noise_channels), dtype=y0.dtype,
                              device=y0.device, levy_area_approximation=levy_area_approximation)

    return sde, y0, ts, bm, method


def integrate(sde, y0, ts, bm, method, dt, adaptive, rtol, atol, dt_min, options, logqp=False):
    if options is None:
        options = {}

    solver_fn = methods.select(method=method, sde_type=sde.sde_type)
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
    if adaptive and method == METHODS.euler and sde.noise_type != NOISE_TYPES.additive:
        warnings.warn(f"Numerical solution is not guaranteed to converge to the correct solution when using adaptive "
                      f"time-stepping with the Euler--Maruyama method with non-additive noise.")

    ys = solver.integrate(ts)

    # --- Backwards compatibility: v0.1.1. ---
    if logqp:
        ys, log_ratio = ys.split(split_size=(y0.size(1) - 1, 1), dim=2)
        log_ratio_increments = torch.stack(
            [log_ratio_t_plus_1 - log_ratio_t
             for log_ratio_t_plus_1, log_ratio_t in zip(log_ratio[1:], log_ratio[:-1])], dim=0
        ).squeeze(dim=2)
        return ys, log_ratio_increments
    # ----------------------------------------

    return solver.integrate(ts)

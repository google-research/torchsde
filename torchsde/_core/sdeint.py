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
from ..types import Any, Dict, Optional, Scalar, Tensor, Tensors, TensorOrTensors, Vector


def sdeint(sde,
           y0: Tensor,
           ts: Vector,
           bm: Optional[BaseBrownian] = None,
           method: Optional[str] = None,
           dt: Scalar = 1e-3,
           adaptive: bool = False,
           rtol: Scalar = 1e-5,
           atol: Scalar = 1e-4,
           dt_min: Scalar = 1e-5,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None,
           logqp: bool = False,
           extra: bool = False,
           extra_solver_state: Optional[Tensors] = None,
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
        extra (bool, optional): If `True`, also return the extra hidden state
            used internally in the solver.
        extra_solver_state: (tuple of Tensors, optional): Additional state to
            initialise the solver with. Some solvers keep track of additional
            state besides y0, and this offers a way to optionally initialise
            that state.

    Returns:
        A single state tensor of size (T, batch_size, d).
        if logqp is True, then the log-ratio penalty is also returned.
        If extra is True, the any extra internal state of the solver is also
        returned.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or if `sde` is missing required methods.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint`")
    del unused_kwargs

    sde, y0, ts, bm, method, options = check_contract(sde, y0, ts, bm, method, adaptive, options, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'atol', 'dt_min'],
                        [ts, dt, rtol, atol, dt_min])

    solver_fn = methods.select(method=method, sde_type=sde.sde_type)
    solver = solver_fn(
        sde=sde,
        bm=bm,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options
    )
    if extra_solver_state is None:
        extra_solver_state = solver.init_extra_solver_state(ts[0], y0)
    ys, extra_solver_state = solver.integrate(y0, ts, extra_solver_state)

    return parse_return(y0, ys, extra_solver_state, extra, logqp)


def check_contract(sde, y0, ts, bm, method, adaptive, options, names, logqp):
    if names is None:
        names_to_change = {}
    else:
        names_to_change = {key: names[key] for key in ("drift", "diffusion", "prior_drift", "drift_and_diffusion",
                                                       "drift_and_diffusion_prod") if key in names}
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)

    if not hasattr(sde, "noise_type"):
        raise ValueError(f"sde does not have the attribute noise_type.")

    if sde.noise_type not in NOISE_TYPES:
        raise ValueError(f"Expected noise type in {NOISE_TYPES}, but found {sde.noise_type}.")

    if not hasattr(sde, "sde_type"):
        raise ValueError(f"sde does not have the attribute sde_type.")

    if sde.sde_type not in SDE_TYPES:
        raise ValueError(f"Expected sde type in {SDE_TYPES}, but found {sde.sde_type}.")

    if not torch.is_tensor(y0):
        raise ValueError("`y0` must be a torch.Tensor.")
    if y0.dim() != 2:
        raise ValueError("`y0` must be a 2-dimensional tensor of shape (batch, channels).")

    # --- Backwards compatibility: v0.1.1. ---
    if logqp:
        sde = base_sde.SDELogqp(sde)
        y0 = torch.cat((y0, y0.new_zeros(size=(y0.size(0), 1))), dim=1)
    # ----------------------------------------

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

    if not torch.is_tensor(ts):
        if not isinstance(ts, (tuple, list)) or not all(isinstance(t, (float, int)) for t in ts):
            raise ValueError(f"Evaluation times `ts` must be a 1-D Tensor or list/tuple of floats.")
        ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)
    if not misc.is_strictly_increasing(ts):
        raise ValueError("Evaluation times `ts` must be strictly increasing.")

    batch_sizes = []
    state_sizes = []
    noise_sizes = []
    batch_sizes.append(y0.size(0))
    state_sizes.append(y0.size(1))
    if bm is not None:
        if len(bm.shape) != 2:
            raise ValueError("`bm` must be of shape (batch, noise_channels).")
        batch_sizes.append(bm.shape[0])
        noise_sizes.append(bm.shape[1])

    def _check_2d(name, shape):
        if len(shape) != 2:
            raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
        batch_sizes.append(shape[0])
        state_sizes.append(shape[1])

    def _check_2d_or_3d(name, shape):
        if sde.noise_type == NOISE_TYPES.diagonal:
            if len(shape) != 2:
                raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[1])
        else:
            if len(shape) != 3:
                raise ValueError(f"{name} must be of shape (batch, state_channels, noise_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[2])

    has_f = False
    has_g = False
    if hasattr(sde, 'f'):
        has_f = True
        f_drift_shape = tuple(sde.f(ts[0], y0).size())
        _check_2d('Drift', f_drift_shape)
    if hasattr(sde, 'g'):
        has_g = True
        g_diffusion_shape = tuple(sde.g(ts[0], y0).size())
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'f_and_g'):
        has_f = True
        has_g = True
        _f, _g = sde.f_and_g(ts[0], y0)
        f_drift_shape = tuple(_f.size())
        g_diffusion_shape = tuple(_g.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'g_prod'):
        has_g = True
        if len(noise_sizes) == 0:
            raise ValueError("Cannot infer noise size (i.e. number of Brownian motion channels). Either pass `bm` "
                             "explicitly, or specify one of the `g`, `f_and_g` functions.`")
        v = torch.randn(batch_sizes[0], noise_sizes[0], dtype=y0.dtype, device=y0.device)
        g_prod_shape = tuple(sde.g_prod(ts[0], y0, v).size())
        _check_2d('Diffusion-vector product', g_prod_shape)
    if hasattr(sde, 'f_and_g_prod'):
        has_f = True
        has_g = True
        if len(noise_sizes) == 0:
            raise ValueError("Cannot infer noise size (i.e. number of Brownian motion channels). Either pass `bm` "
                             "explicitly, or specify one of the `g`, `f_and_g` functions.`")
        v = torch.randn(batch_sizes[0], noise_sizes[0], dtype=y0.dtype, device=y0.device)
        _f, _g_prod = sde.f_and_g_prod(ts[0], y0, v)
        f_drift_shape = tuple(_f.size())
        g_prod_shape = tuple(_g_prod.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d('Diffusion-vector product', g_prod_shape)

    if not has_f:
        raise ValueError("sde must define at least one of `f`, `f_and_g`, or `f_and_g_prod`. (Or possibly more "
                         "depending on the method chosen.)")
    if not has_g:
        raise ValueError("sde must define at least one of `g`, `f_and_g`, `g_prod` or `f_and_g_prod`. (Or possibly "
                         "more depending on the method chosen.)")

    for batch_size in batch_sizes[1:]:
        if batch_size != batch_sizes[0]:
            raise ValueError("Batch sizes not consistent.")
    for state_size in state_sizes[1:]:
        if state_size != state_sizes[0]:
            raise ValueError("State sizes not consistent.")
    for noise_size in noise_sizes[1:]:
        if noise_size != noise_sizes[0]:
            raise ValueError("Noise sizes not consistent.")

    if sde.noise_type == NOISE_TYPES.scalar:
        if noise_sizes[0] != 1:
            raise ValueError(f"Scalar noise must have only one channel; the diffusion has {noise_sizes[0]} noise "
                             f"channels.")

    sde = base_sde.ForwardSDE(sde)

    if bm is None:
        if method == METHODS.srk:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.space_time
        elif method == METHODS.log_ode_midpoint:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.foster
        else:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_sizes[0], noise_sizes[0]), dtype=y0.dtype,
                              device=y0.device, levy_area_approximation=levy_area_approximation)

    if options is None:
        options = {}
    else:
        options = options.copy()

    if adaptive and method == METHODS.euler and sde.noise_type != NOISE_TYPES.additive:
        warnings.warn(f"Numerical solution is not guaranteed to converge to the correct solution when using adaptive "
                      f"time-stepping with the Euler--Maruyama method with non-additive noise.")

    return sde, y0, ts, bm, method, options


def parse_return(y0, ys, extra_solver_state, extra, logqp):
    if logqp:
        ys, log_ratio = ys.split(split_size=(y0.size(1) - 1, 1), dim=2)
        log_ratio_increments = torch.stack(
            [log_ratio_t_plus_1 - log_ratio_t
             for log_ratio_t_plus_1, log_ratio_t in zip(log_ratio[1:], log_ratio[:-1])], dim=0
        ).squeeze(dim=2)

        if extra:
            return ys, log_ratio_increments, extra_solver_state
        else:
            return ys, log_ratio_increments
    else:
        if extra:
            return ys, extra_solver_state
        else:
            return ys

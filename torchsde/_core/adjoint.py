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

import torch
from torch import nn

from . import base_sde
from . import misc
from . import sdeint
from .adjoint_sde import AdjointSDE
from .._brownian import BaseBrownian, ReverseBrownian
from ..settings import METHODS, SDE_TYPES, NOISE_TYPES
from ..types import Scalar, Vector, Optional, Dict, Any, Tensor


class _SdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sde, ts, dt, bm, method, adjoint_method, adaptive, adjoint_adaptive, rtol,  # noqa
                adjoint_rtol, atol, adjoint_atol, dt_min, options, adjoint_options, y0, *params):
        ctx.sde = sde
        ctx.dt = dt
        ctx.bm = bm
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_adaptive = adjoint_adaptive
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.dt_min = dt_min
        ctx.adjoint_options = adjoint_options

        ys = sdeint.integrate(
            sde=sde,
            y0=y0.detach(),  # This .detach() is VERY IMPORTANT. See adjoint_sde.py::AdjointSDE._unpack_y_aug.
            ts=ts,
            bm=bm,
            method=method,
            dt=dt,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            dt_min=dt_min,
            options=options,
        )
        ctx.save_for_backward(ys, ts, *params)
        return ys

    @staticmethod
    def backward(ctx, grad_ys):  # noqa
        ys, ts, *params = ctx.saved_tensors
        sde = ctx.sde
        dt = ctx.dt
        bm = ctx.bm
        adjoint_method = ctx.adjoint_method
        adjoint_adaptive = ctx.adjoint_adaptive
        adjoint_rtol = ctx.adjoint_rtol
        adjoint_atol = ctx.adjoint_atol
        dt_min = ctx.dt_min
        adjoint_options = ctx.adjoint_options

        aug_state = [ys[-1], grad_ys[-1]] + [torch.zeros_like(param) for param in params]
        shapes = [t.size() for t in aug_state]
        adjoint_sde = AdjointSDE(sde, params, shapes)
        reverse_bm = ReverseBrownian(bm)

        for i in range(ys.size(0) - 1, 0, -1):
            aug_state = misc.flatten(aug_state)
            aug_state = _SdeintAdjointMethod.apply(adjoint_sde, torch.stack([-ts[i], -ts[i - 1]]), dt, reverse_bm,
                                                   adjoint_method, adjoint_method, adjoint_adaptive, adjoint_adaptive,
                                                   adjoint_rtol, adjoint_rtol, adjoint_atol, adjoint_atol, dt_min,
                                                   adjoint_options, adjoint_options, aug_state, *params)
            aug_state = misc.flat_to_shape(aug_state[1], shapes)  # Unpack the state at time -ts[i - 1].
            aug_state[0] = ys[i - 1]
            aug_state[1] = aug_state[1] + grad_ys[i - 1]

        return (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, *aug_state[1:]
        )


def sdeint_adjoint(sde: nn.Module,
                   y0: Tensor,
                   ts: Vector,
                   bm: Optional[BaseBrownian] = None,
                   method: Optional[str] = "srk",
                   adjoint_method: Optional[str] = None,
                   dt: Optional[Scalar] = 1e-3,
                   adaptive: Optional[bool] = False,
                   adjoint_adaptive: Optional[bool] = False,
                   rtol: Optional[Scalar] = 1e-5,
                   adjoint_rtol: Optional[Scalar] = 1e-5,
                   atol: Optional[Scalar] = 1e-4,
                   adjoint_atol: Optional[Scalar] = 1e-4,
                   dt_min: Optional[Scalar] = 1e-5,
                   options: Optional[Dict[str, Any]] = None,
                   adjoint_options: Optional[Dict[str, Any]] = None,
                   names: Optional[Dict[str, str]] = None,
                   **unused_kwargs) -> Tensor:
    """Numerically integrate an Itô SDE with stochastic adjoint support.

    Args:
        sde (torch.nn.Module): Object with methods `f` and `g` representing the
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
        method (str, optional): Name of numerical integration method.
        adjoint_method (str, optional): Name of numerical integration method for
            backward adjoint solve. Defaults to a sensible choice depending on
            the noise type of the supplied SDE.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        adjoint_adaptive (bool, optional): If `True`, use adaptive time-stepping
            for the backward adjoint solve.
        rtol (float, optional): Relative tolerance.
        adjoint_rtol (float, optional): Relative tolerance for backward adjoint
            solve.
        atol (float, optional): Absolute tolerance.
        adjoint_atol (float, optional): Absolute tolerance for backward adjoint
            solve.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        adjoint_options (dict, optional): Dict of options for the integration
            method of the backward adjoint solve.
        names (dict, optional): Dict of method names for drift and diffusion.
            Expected keys are "drift" and "diffusion". Serves so that users can
            use methods with names not in `("f", "g")`, e.g. to use the
            method "foo" for the drift, we supply `names={"drift": "foo"}`.

    Returns:
        A single state tensor of size (T, batch_size, d).

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or `sde` is missing required methods.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint_adjoint`")
    del unused_kwargs

    if not isinstance(sde, nn.Module):
        raise ValueError("`sde` is required to be an instance of nn.Module.")

    sde, y0, ts, bm = sdeint.check_contract(sde, y0, ts, bm, method, names)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'adjoint_rtol', 'atol', 'adjoint_atol', 'dt_min'],
                        [ts, dt, rtol, adjoint_rtol, atol, adjoint_atol, dt_min])
    adjoint_method = _select_default_adjoint_method(sde, adjoint_method)
    params = filter(lambda x: x.requires_grad, sde.parameters())

    return _SdeintAdjointMethod.apply(  # noqa
        sde, ts, dt, bm, method, adjoint_method, adaptive, adjoint_adaptive, rtol, adjoint_rtol, atol,
        adjoint_atol, dt_min, options, adjoint_options, y0, *params
    )


def _select_default_adjoint_method(sde: base_sde.ForwardSDE, adjoint_method: str) -> str:
    sde_type, noise_type = sde.sde_type, sde.noise_type

    if adjoint_method is None:  # Select the default based on noise type of forward.
        adjoint_method = {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: METHODS.milstein,
                NOISE_TYPES.additive: METHODS.euler,
                NOISE_TYPES.scalar: METHODS.euler,
            }.get(noise_type, "unsupported"),
            SDE_TYPES.stratonovich: {
                NOISE_TYPES.general: METHODS.midpoint,
            }.get(noise_type, "unsupported")
        }[sde_type]

        if adjoint_method == "unsupported":
            raise ValueError(f"Adjoint not supported for {sde_type} SDEs with noise type {noise_type}.")

    return adjoint_method

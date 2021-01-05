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
from ..settings import METHODS, NOISE_TYPES, SDE_TYPES
from ..types import Any, Dict, Optional, Scalar, Tensor, TensorOrTensors, Vector


class _SdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sde, ts, dt, bm, method, adjoint_method, adaptive, adjoint_adaptive, rtol,  # noqa
                adjoint_rtol, atol, adjoint_atol, dt_min, options, adjoint_options, y0, *adjoint_params):
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
            y0=y0.detach(),  # This .detach() is VERY IMPORTANT. See adjoint_sde.py::AdjointSDE._get_state.
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
        ctx.save_for_backward(ys, ts, *adjoint_params)
        return ys

    @staticmethod
    def backward(ctx, grad_ys):  # noqa
        ys, ts, *adjoint_params = ctx.saved_tensors
        sde = ctx.sde
        dt = ctx.dt
        bm = ctx.bm
        adjoint_method = ctx.adjoint_method
        adjoint_adaptive = ctx.adjoint_adaptive
        adjoint_rtol = ctx.adjoint_rtol
        adjoint_atol = ctx.adjoint_atol
        dt_min = ctx.dt_min
        adjoint_options = ctx.adjoint_options

        aug_state = [ys[-1], grad_ys[-1]] + [torch.zeros_like(param) for param in adjoint_params]
        shapes = [t.size() for t in aug_state]
        adjoint_sde = AdjointSDE(sde, adjoint_params, shapes)
        reverse_bm = ReverseBrownian(bm)

        for i in range(ys.size(0) - 1, 0, -1):
            aug_state = misc.flatten(aug_state)
            aug_state = _SdeintAdjointMethod.apply(adjoint_sde,
                                                   torch.stack([-ts[i], -ts[i - 1]]),
                                                   dt,
                                                   reverse_bm,
                                                   adjoint_method,
                                                   adjoint_method,
                                                   adjoint_adaptive,
                                                   adjoint_adaptive,
                                                   adjoint_rtol,
                                                   adjoint_rtol,
                                                   adjoint_atol,
                                                   adjoint_atol,
                                                   dt_min,
                                                   adjoint_options,
                                                   adjoint_options,
                                                   aug_state,
                                                   *adjoint_params)
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
                   method: Optional[str] = None,
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
                   adjoint_params=None,
                   names: Optional[Dict[str, str]] = None,
                   logqp: Optional[bool] = False,
                   **unused_kwargs) -> TensorOrTensors:
    """Numerically integrate an SDE with stochastic adjoint support.

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
        method (str, optional): Numerical integration method to use. Must be
            compatible with the SDE type (Ito/Stratonovich) and the noise type
            (scalar/additive/diagonal/general). Defaults to a sensible choice
            depending on the SDE type and noise type of the supplied SDE.
        adjoint_method (str, optional): Name of numerical integration method for
            backward adjoint solve. Defaults to a sensible choice depending on
            the SDE type and noise type of the supplied SDE.
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
        adjoint_params (Sequence of Tensors, optional): Tensors whose gradient
            should be obtained with the adjoint. If not specified, defaults to
            the parameters of `sde`.
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
            or `sde` is missing required methods.

    Note:
        The backward pass is much more efficient with Stratonovich SDEs than
        with Ito SDEs.

    Note:
        Double-backward is supported for Stratonovich SDEs. Doing so will use
        the adjoint method to compute the gradient of the adjoint. (i.e. rather
        than backpropagating through the numerical solver used for the
        adjoint.) The same `adjoint_method`, `adjoint_adaptive`, `adjoint_rtol,
        `adjoint_atol`, `adjoint_options` will be used for the second-order
        adjoint as is used for the first-order adjoint.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint_adjoint`")
    del unused_kwargs

    if adjoint_params is None and not isinstance(sde, nn.Module):
        raise ValueError('`sde` must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    sde, y0, ts, bm, method = sdeint.check_contract(sde, y0, ts, bm, method, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'adjoint_rtol', 'atol', 'adjoint_atol', 'dt_min'],
                        [ts, dt, rtol, adjoint_rtol, atol, adjoint_atol, dt_min])
    adjoint_params = tuple(sde.parameters()) if adjoint_params is None else tuple(adjoint_params)
    adjoint_params = filter(lambda x: x.requires_grad, adjoint_params)
    adjoint_method = _select_default_adjoint_method(sde, adjoint_method)

    ys = _SdeintAdjointMethod.apply(  # noqa
        sde, ts, dt, bm, method, adjoint_method, adaptive, adjoint_adaptive, rtol, adjoint_rtol, atol,
        adjoint_atol, dt_min, options, adjoint_options, y0, *adjoint_params
    )

    # --- Backwards compatibility: v0.1.1. ---
    if logqp:
        ys, log_ratio = ys.split(split_size=(y0.size(dim=1) - 1, 1), dim=2)
        log_ratio_increments = torch.stack(
            [log_ratio_t_plus_1 - log_ratio_t
             for log_ratio_t_plus_1, log_ratio_t in zip(log_ratio[1:], log_ratio[:-1])], dim=0
        ).squeeze(dim=2)
        return ys, log_ratio_increments
    # ----------------------------------------

    return ys


def _select_default_adjoint_method(sde: base_sde.ForwardSDE, adjoint_method: str) -> str:
    """Select the default method for adjoint computation based on the noise type of the forward SDE."""
    if adjoint_method is None:
        adjoint_method = {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: METHODS.milstein,
                NOISE_TYPES.additive: METHODS.euler,
                NOISE_TYPES.scalar: METHODS.euler,
                NOISE_TYPES.general: METHODS.euler,
            }[sde.noise_type],
            SDE_TYPES.stratonovich: METHODS.midpoint,
        }[sde.sde_type]
    return adjoint_method

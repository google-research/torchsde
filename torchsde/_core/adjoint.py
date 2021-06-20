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
import warnings

from . import base_sde
from . import methods
from . import misc
from . import sdeint
from .adjoint_sde import AdjointSDE
from .._brownian import BaseBrownian, ReverseBrownian
from ..settings import METHODS, NOISE_TYPES, SDE_TYPES
from ..types import Any, Dict, Optional, Scalar, Tensor, Tensors, TensorOrTensors, Vector


class _SdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sde, ts, dt, bm, solver, method, adjoint_method, adjoint_adaptive, adjoint_rtol, adjoint_atol,
                dt_min, adjoint_options, len_extras, y0, *extras_and_adjoint_params):
        ctx.sde = sde
        ctx.dt = dt
        ctx.bm = bm
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_adaptive = adjoint_adaptive
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.dt_min = dt_min
        ctx.adjoint_options = adjoint_options
        ctx.len_extras = len_extras

        extra_solver_state = extras_and_adjoint_params[:len_extras]
        adjoint_params = extras_and_adjoint_params[len_extras:]

        # This .detach() is VERY IMPORTANT. See adjoint_sde.py::AdjointSDE.get_state.
        y0 = y0.detach()
        # Necessary for the same reason
        extra_solver_state = tuple(x.detach() for x in extra_solver_state)
        ys, extra_solver_state = solver.integrate(y0, ts, extra_solver_state)

        if method == METHODS.reversible_heun and adjoint_method == METHODS.adjoint_reversible_heun:
            ctx.saved_extras_for_backward = True
            extras_for_backward = extra_solver_state
        else:
            # Else just remove the `extra_solver_state` information.
            ctx.saved_extras_for_backward = False
            extras_for_backward = ()
        ctx.save_for_backward(ys, ts, *extras_for_backward, *adjoint_params)
        return (ys, *extra_solver_state)

    @staticmethod
    def backward(ctx, grad_ys, *grad_extra_solver_state):  # noqa
        ys, ts, *extras_and_adjoint_params = ctx.saved_tensors
        if ctx.saved_extras_for_backward:
            extra_solver_state = extras_and_adjoint_params[:ctx.len_extras]
            adjoint_params = extras_and_adjoint_params[ctx.len_extras:]
        else:
            grad_extra_solver_state = ()
            extra_solver_state = None
            adjoint_params = extras_and_adjoint_params

        aug_state = [ys[-1], grad_ys[-1]] + list(grad_extra_solver_state) + [torch.zeros_like(param)
                                                                             for param in adjoint_params]
        shapes = [t.size() for t in aug_state]
        aug_state = misc.flatten(aug_state)
        aug_state = aug_state.unsqueeze(0)  # dummy batch dimension
        adjoint_sde = AdjointSDE(ctx.sde, adjoint_params, shapes)
        reverse_bm = ReverseBrownian(ctx.bm)

        solver_fn = methods.select(method=ctx.adjoint_method, sde_type=adjoint_sde.sde_type)
        solver = solver_fn(
            sde=adjoint_sde,
            bm=reverse_bm,
            dt=ctx.dt,
            adaptive=ctx.adjoint_adaptive,
            rtol=ctx.adjoint_rtol,
            atol=ctx.adjoint_atol,
            dt_min=ctx.dt_min,
            options=ctx.adjoint_options
        )
        if extra_solver_state is None:
            extra_solver_state = solver.init_extra_solver_state(ts[-1], aug_state)

        for i in range(ys.size(0) - 1, 0, -1):
            (_, aug_state), *extra_solver_state = _SdeintAdjointMethod.apply(adjoint_sde,
                                                                             torch.stack([-ts[i], -ts[i - 1]]),
                                                                             ctx.dt,
                                                                             reverse_bm,
                                                                             solver,
                                                                             ctx.adjoint_method,
                                                                             ctx.adjoint_method,
                                                                             ctx.adjoint_adaptive,
                                                                             ctx.adjoint_rtol,
                                                                             ctx.adjoint_atol,
                                                                             ctx.dt_min,
                                                                             ctx.adjoint_options,
                                                                             len(extra_solver_state),
                                                                             aug_state,
                                                                             *extra_solver_state,
                                                                             *adjoint_params)
            aug_state = misc.flat_to_shape(aug_state.squeeze(0), shapes)
            aug_state[0] = ys[i - 1]
            aug_state[1] = aug_state[1] + grad_ys[i - 1]
            if i != 1:
                aug_state = misc.flatten(aug_state)
                aug_state = aug_state.unsqueeze(0)  # dummy batch dimension

        if ctx.saved_extras_for_backward:
            out = aug_state[1:]
        else:
            out = [aug_state[1]] + ([None] * ctx.len_extras) + aug_state[2:]
        return (
            None, None, None, None, None, None, None, None, None, None, None, None, None, *out,
        )


def sdeint_adjoint(sde: nn.Module,
                   y0: Tensor,
                   ts: Vector,
                   bm: Optional[BaseBrownian] = None,
                   method: Optional[str] = None,
                   adjoint_method: Optional[str] = None,
                   dt: Scalar = 1e-3,
                   adaptive: bool = False,
                   adjoint_adaptive: bool = False,
                   rtol: Scalar = 1e-5,
                   adjoint_rtol: Scalar = 1e-5,
                   atol: Scalar = 1e-4,
                   adjoint_atol: Scalar = 1e-4,
                   dt_min: Scalar = 1e-5,
                   options: Optional[Dict[str, Any]] = None,
                   adjoint_options: Optional[Dict[str, Any]] = None,
                   adjoint_params=None,
                   names: Optional[Dict[str, str]] = None,
                   logqp: bool = False,
                   extra: bool = False,
                   extra_solver_state: Optional[Tensors] = None,
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

    sde, y0, ts, bm, method, options = sdeint.check_contract(sde, y0, ts, bm, method, adaptive, options, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'adjoint_rtol', 'atol', 'adjoint_atol', 'dt_min'],
                        [ts, dt, rtol, adjoint_rtol, atol, adjoint_atol, dt_min])
    adjoint_params = tuple(sde.parameters()) if adjoint_params is None else tuple(adjoint_params)
    adjoint_params = filter(lambda x: x.requires_grad, adjoint_params)
    adjoint_method = _select_default_adjoint_method(sde, method, adjoint_method)
    adjoint_options = {} if adjoint_options is None else adjoint_options.copy()

    # Note that all of these warnings are only applicable for reversible solvers with sdeint_adjoint; none of them
    # apply to sdeint.
    if method == METHODS.reversible_heun:
        if adjoint_method != METHODS.adjoint_reversible_heun:
            warnings.warn(f"method={repr(method)}, but adjoint_method!={repr(METHODS.adjoint_reversible_heun)}.")
        if adaptive or adjoint_adaptive:
            warnings.warn(f"A limitation of the current method={repr(method)} implementation is "
                          f"that it does not save the time steps used. This means that it may not be perfectly "
                          f"accurate when used with `adaptive` or `adjoint_adaptive`.")
        else:
            num_steps = (ts - ts[0]) / dt
            if not torch.allclose(num_steps, num_steps.round()):
                warnings.warn(f"The spacing between time points `ts` is not an integer multiple of the time step `dt`. "
                              f"This means that the backward pass (which is forced to step to each of `ts` to get "
                              f"dL/dy(t) for t in ts) will not perfectly mimick the forward pass (which does not step "
                              f"to each `ts`, and instead interpolates to them). This means that "
                              f"method={repr(method)} may not be perfectly accurate.")

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

    ys, *extra_solver_state = _SdeintAdjointMethod.apply(
        sde, ts, dt, bm, solver, method, adjoint_method, adjoint_adaptive, adjoint_rtol, adjoint_atol, dt_min,
        adjoint_options, len(extra_solver_state), y0, *extra_solver_state, *adjoint_params
    )

    return sdeint.parse_return(y0, ys, extra_solver_state, extra, logqp)


def _select_default_adjoint_method(sde: base_sde.ForwardSDE, method: str, adjoint_method: Optional[str]) -> str:
    """Select the default method for adjoint computation based on the noise type of the forward SDE."""
    if adjoint_method is not None:
        return adjoint_method
    elif method == METHODS.reversible_heun:
        return METHODS.adjoint_reversible_heun
    else:
        return {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: METHODS.milstein,
                NOISE_TYPES.additive: METHODS.euler,
                NOISE_TYPES.scalar: METHODS.euler,
                NOISE_TYPES.general: METHODS.euler,
            }[sde.noise_type],
            SDE_TYPES.stratonovich: METHODS.midpoint,
        }[sde.sde_type]

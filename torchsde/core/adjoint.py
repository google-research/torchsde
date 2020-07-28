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

from typing import Tuple, Union, Optional, List, Dict, Any

import torch
from torch import nn

try:
    from torchsde.brownian_lib import BrownianPath
except Exception:
    from torchsde.brownian.brownian_path import BrownianPath

from torchsde.brownian import base
from torchsde.core import base_sde
from torchsde.core import methods
from torchsde.core import misc
from torchsde.core import sdeint


class _SdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 14, 'Internal error: all arguments required.'
        y0 = args[:-13]
        (sde, ts, flat_params, dt, bm, method, adjoint_method, adaptive, rtol, atol, dt_min, options,
         adjoint_options) = args[-13:]
        (ctx.sde, ctx.dt, ctx.bm, ctx.adjoint_method, ctx.adaptive, ctx.rtol, ctx.atol, ctx.dt_min,
         ctx.adjoint_options) = sde, dt, bm, adjoint_method, adaptive, rtol, atol, dt_min, adjoint_options

        sde = base_sde.ForwardSDEIto(sde)
        ans = sdeint.integrate(
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
            options=options
        )
        ctx.save_for_backward(ts, flat_params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_outputs):
        ts, flat_params, *ans = ctx.saved_tensors
        sde, dt, bm, adjoint_method, adaptive, rtol, atol, dt_min, adjoint_options = (
            ctx.sde, ctx.dt, ctx.bm, ctx.adjoint_method, ctx.adaptive, ctx.rtol, ctx.atol, ctx.dt_min,
            ctx.adjoint_options
        )
        params = misc.make_seq_requires_grad(sde.parameters())
        n_tensors, n_params = len(ans), len(params)

        # TODO: Make use of adjoint_method.
        aug_bm = lambda t: tuple(-bmi for bmi in bm(-t))
        adjoint_sde, adjoint_method, adjoint_adaptive = _get_adjoint_params(sde=sde, params=params, adaptive=adaptive)

        T = ans[0].size(0)
        adj_y = tuple(grad_outputs_[-1] for grad_outputs_ in grad_outputs)
        adj_params = torch.zeros_like(flat_params)

        for i in range(T - 1, 0, -1):
            ans_i = tuple(ans_[i] for ans_ in ans)
            aug_y0 = (*ans_i, *adj_y, adj_params)

            aug_ans = sdeint.integrate(
                sde=adjoint_sde,
                y0=aug_y0,
                ts=torch.tensor([-ts[i], -ts[i - 1]]).to(ts),
                bm=aug_bm,
                method=adjoint_method,
                dt=dt,
                adaptive=adjoint_adaptive,
                rtol=rtol,
                atol=atol,
                dt_min=dt_min,
                options=adjoint_options
            )

            adj_y = aug_ans[n_tensors:2 * n_tensors]
            adj_params = aug_ans[-1]

            adj_y = tuple(adj_y_[1] for adj_y_ in adj_y)
            adj_params = adj_params[1]

            adj_y = misc.seq_add(adj_y, tuple(grad_outputs_[i - 1] for grad_outputs_ in grad_outputs))

            del aug_y0, aug_ans

        return (*adj_y, None, None, adj_params, None, None, None, None, None, None, None, None, None, None)


class _SdeintLogqpAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 14, 'Internal error: all arguments required.'
        y0 = args[:-13]
        (sde, ts, flat_params, dt, bm, method, adjoint_method, adaptive, rtol, atol, dt_min, options,
         adjoint_options) = args[-13:]
        (ctx.sde, ctx.dt, ctx.bm, ctx.adjoint_method, ctx.adaptive, ctx.rtol, ctx.atol, ctx.dt_min,
         ctx.adjoint_options) = sde, dt, bm, adjoint_method, adaptive, rtol, atol, dt_min, adjoint_options

        sde = base_sde.ForwardSDEIto(sde)
        ans_and_logqp = sdeint.integrate(
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
            logqp=True
        )
        ans, logqp = ans_and_logqp[:len(y0)], ans_and_logqp[len(y0):]

        # Don't need to save `logqp`, since it is never used in the backward pass to compute gradients.
        ctx.save_for_backward(ts, flat_params, *ans)
        return ans + logqp

    @staticmethod
    def backward(ctx, *grad_outputs):
        ts, flat_params, *ans = ctx.saved_tensors
        sde, dt, bm, adjoint_method, adaptive, rtol, atol, dt_min, adjoint_options = (
            ctx.sde, ctx.dt, ctx.bm, ctx.adjoint_method, ctx.adaptive, ctx.rtol, ctx.atol, ctx.dt_min,
            ctx.adjoint_options
        )
        params = misc.make_seq_requires_grad(sde.parameters())
        n_tensors, n_params = len(ans), len(params)

        # TODO: Make use of adjoint_method.
        aug_bm = lambda t: tuple(-bmi for bmi in bm(-t))
        adjoint_sde, adjoint_method, adjoint_adaptive = _get_adjoint_params(
            sde=sde, params=params, adaptive=adaptive, logqp=True)

        T = ans[0].size(0)
        adj_y = tuple(grad_outputs_[-1] for grad_outputs_ in grad_outputs[:n_tensors])
        adj_l = tuple(grad_outputs_[-1] for grad_outputs_ in grad_outputs[n_tensors:])
        adj_params = torch.zeros_like(flat_params)

        for i in range(T - 1, 0, -1):
            ans_i = tuple(ans_[i] for ans_ in ans)
            aug_y0 = (*ans_i, *adj_y, *adj_l, adj_params)

            aug_ans = sdeint.integrate(
                sde=adjoint_sde,
                y0=aug_y0,
                ts=torch.tensor([-ts[i], -ts[i - 1]]).to(ts),
                bm=aug_bm,
                method=adjoint_method,
                dt=dt,
                adaptive=adjoint_adaptive,
                rtol=rtol,
                atol=atol,
                dt_min=dt_min,
                options=adjoint_options
            )

            adj_y = aug_ans[n_tensors:2 * n_tensors]
            adj_params = aug_ans[-1]

            adj_y = tuple(adj_y_[1] for adj_y_ in adj_y)
            adj_params = adj_params[1]

            adj_y = misc.seq_add(adj_y, tuple(grad_outputs_[i - 1] for grad_outputs_ in grad_outputs[:n_tensors]))
            adj_l = tuple(grad_outputs_[i - 1] for grad_outputs_ in grad_outputs[n_tensors:])

            del aug_y0, aug_ans

        return (*adj_y, None, None, adj_params, None, None, None, None, None, None, None, None, None, None)


def sdeint_adjoint(sde,
                   y0: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
                   ts: Union[torch.Tensor, Tuple[float, ...], List[float]],
                   bm: Optional[base.Brownian] = None,
                   logqp: Optional[bool] = False,
                   method: Optional[str] = 'srk',
                   adjoint_method: Optional[str] = 'milstein',
                   dt: Optional[float] = 1e-3,
                   adaptive: Optional[bool] = False,
                   rtol: Optional[float] = 1e-6,
                   atol: Optional[float] = 1e-5,
                   dt_min: Optional[float] = 1e-4,
                   options: Optional[Dict[str, Any]] = None,
                   adjoint_options: Optional[Dict[str, Any]] = None,
                   names: Optional[Dict[str, str]] = None):
    """Numerically integrate an Itô SDE with stochastic adjoint support.

    Args:
        sde: An object with the methods `f` and `g` representing the drift and
            diffusion functions. The output of `g` should either be a single
            (or a tuple of) tensor(s) of size (batch_size, d) for diagonal noise
            SDEs or (batch_size, d, m) for SDEs of other noise type, where d is
            the dimensionality of state and m is the dimensionality of the
            Brownian motion.
        y0: A single (or a tuple of) tensor(s) of size (batch_size, d).
        ts: A list or size (T,) tensor in non-descending order.
        bm: A `BrownianPath` or `BrownianTree` object.
            Defaults to `BrownianPath` for diagonal noise residing on CPU.
        logqp: If True, also return the log-ratio penalty across the whole path.
        method: Numerical integration method.
        adjoint_method: Numerical integration method of backward adjoint solve.
        dt: The constant step size or initial step size for adaptive stepping.
        adaptive: If True, use adaptive time-stepping.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        dt_min: Minimum step size for adaptive stepping.
        options: Optional dict of options for the indicated integration method.
        adjoint_options: Optional dict of options for the indicated integration
            method of the backward adjoint solve.
        names: Optional dict of method names for drift, diffusion, and prior
            drift. Expected keys are "drift", "diffusion", and "prior_drift".

    Returns:
        A single state tensor of size (T, batch_size, d) or a tuple of such
        tensors. Also returns a single log-ratio tensor of size
        (T - 1, batch_size) or a tuple of such tensors, if `logqp`=True.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or `sde` is missing required methods.
    """
    if not isinstance(sde, nn.Module):
        raise ValueError('sde is required to be an instance of nn.Module.')

    names_to_change = sdeint.get_names_to_change(names)
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)
    sdeint.check_contract(sde=sde, method=method, adaptive=adaptive, logqp=logqp, adjoint_method=adjoint_method)

    if bm is None:
        bm = BrownianPath(t0=ts[0], w0=torch.zeros_like(y0).cpu())

    tensor_input = isinstance(y0, torch.Tensor)
    if tensor_input:
        sde = base_sde.TupleSDE(sde)
        y0 = (y0,)
        bm_ = bm
        bm = lambda t: (bm_(t),)

    flat_params = misc.flatten(sde.parameters())
    if logqp:
        return _SdeintLogqpAdjointMethod.apply(
            *y0, sde, ts, flat_params, dt, bm, method, adjoint_method, adaptive, rtol, atol, dt_min,
            options, adjoint_options
        )

    ys = _SdeintAdjointMethod.apply(
        *y0, sde, ts, flat_params, dt, bm, method, adjoint_method, adaptive, rtol, atol, dt_min,
        options, adjoint_options
    )
    return ys[0] if tensor_input else ys


def _get_adjoint_params(sde, params, adaptive, logqp=False):
    if sde.noise_type == "diagonal":
        if logqp:
            adjoint_sde = methods.AdjointSDEDiagonalLogqp(sde, params=params)
        else:
            adjoint_sde = methods.AdjointSDEDiagonal(sde, params=params)

        adjoint_method = "milstein"

    elif sde.noise_type == "scalar":
        if logqp:
            adjoint_sde = methods.AdjointSDEScalarLogqp(sde, params=params)
        else:
            adjoint_sde = methods.AdjointSDEScalar(sde, params=params)

        adjoint_method = "euler"

    elif sde.noise_type == "additive":
        if logqp:
            adjoint_sde = methods.AdjointSDEAdditiveLogqp(sde, params=params)
        else:
            adjoint_sde = methods.AdjointSDEAdditive(sde, params=params)

        adjoint_method = "euler"

    else:
        raise ValueError('Adjoint mode for general noise SDEs not supported.')

    if adjoint_method in ('milstein', 'srk') and adaptive:  # Need method with strong order >= 1.0.
        adjoint_adaptive = True
    else:
        adjoint_adaptive = False

    return adjoint_sde, adjoint_method, adjoint_adaptive

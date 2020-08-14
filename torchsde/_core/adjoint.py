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

from typing import Optional, Dict, Any

import torch
from torch import nn

try:
    from torchsde.brownian_lib import BrownianPath
except Exception:  # noqa
    from torchsde._brownian.brownian_path import BrownianPath  # noqa

from torchsde._brownian.base_brownian import Brownian  # noqa
from torchsde._core import base_sde
from torchsde._core import methods
from torchsde._core import misc
from torchsde._core.types import TensorOrTensors, Scalar, Vector
from torchsde._core import sdeint


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

        # TODO: Make use of `adjoint_method`.
        aug_bm = lambda ta, tb: [-bmi for bmi in bm(-tb, -ta)]  # noqa
        adjoint_sde, adjoint_method, adjoint_adaptive = _get_adjoint_params(
            sde=sde, params=params, adaptive=adaptive
        )

        T = ans[0].size(0)
        adj_y = [grad_outputs_[-1] for grad_outputs_ in grad_outputs]
        adj_params = torch.zeros_like(flat_params)

        for i in range(T - 1, 0, -1):
            ans_i = [ans_[i] for ans_ in ans]
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

            adj_y = [adj_y_[1] for adj_y_ in adj_y]
            adj_params = adj_params[1]

            adj_y = misc.seq_add(adj_y, [grad_outputs_[i - 1] for grad_outputs_ in grad_outputs])

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

        # TODO: Make use of `adjoint_method`.
        aug_bm = lambda ta, tb: [-bmi for bmi in bm(-tb, -ta)]  # noqa
        adjoint_sde, adjoint_method, adjoint_adaptive = _get_adjoint_params(
            sde=sde, params=params, adaptive=adaptive, logqp=True
        )

        T = ans[0].size(0)
        adj_y = [grad_outputs_[-1] for grad_outputs_ in grad_outputs[:n_tensors]]
        adj_l = [grad_outputs_[-1] for grad_outputs_ in grad_outputs[n_tensors:]]
        adj_params = torch.zeros_like(flat_params)

        for i in range(T - 1, 0, -1):
            ans_i = [ans_[i] for ans_ in ans]
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

            adj_y = [adj_y_[1] for adj_y_ in adj_y]
            adj_params = adj_params[1]

            adj_y = misc.seq_add(adj_y, [grad_outputs_[i - 1] for grad_outputs_ in grad_outputs[:n_tensors]])
            adj_l = [grad_outputs_[i - 1] for grad_outputs_ in grad_outputs[n_tensors:]]

            del aug_y0, aug_ans

        return (*adj_y, None, None, adj_params, None, None, None, None, None, None, None, None, None, None)


def sdeint_adjoint(sde,
                   y0: TensorOrTensors,
                   ts: Vector,
                   bm: Optional[Brownian] = None,
                   logqp: Optional[bool] = False,
                   method: Optional[str] = 'srk',
                   adjoint_method: Optional[str] = 'milstein',
                   dt: Optional[Scalar] = 1e-3,
                   adaptive: Optional[bool] = False,
                   rtol: Optional[float] = 1e-5,
                   atol: Optional[float] = 1e-4,
                   dt_min: Optional[Scalar] = 1e-5,
                   options: Optional[Dict[str, Any]] = None,
                   adjoint_options: Optional[Dict[str, Any]] = None,
                   names: Optional[Dict[str, str]] = None):
    """Numerically integrate an Itô SDE with stochastic adjoint support.

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
        adjoint_method (str, optional): Name of numerical integration method for
            backward adjoint solve.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        rtol (float, optional): Relative tolerance.
        atol (float, optional): Absolute tolerance.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        adjoint_options (dict, optional): Dict of options for the integration
            method of the backward adjoint solve.
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
    if not isinstance(sde, nn.Module):
        raise ValueError('sde is required to be an instance of nn.Module.')

    names_to_change = sdeint.get_names_to_change(names)
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)
    sdeint.check_contract(sde=sde, method=method, logqp=logqp, adjoint_method=adjoint_method)

    if bm is None:
        bm = BrownianPath(t0=ts[0], w0=torch.zeros_like(y0).cpu())

    tensor_input = isinstance(y0, torch.Tensor)
    if tensor_input:
        sde = base_sde.TupleSDE(sde)
        y0 = (y0,)
        bm_ = bm
        bm = lambda t: (bm_(t),)  # noqa

    flat_params = misc.flatten(sde.parameters())
    if logqp:
        return _SdeintLogqpAdjointMethod.apply(  # noqa
            *y0, sde, ts, flat_params, dt, bm, method, adjoint_method, adaptive, rtol, atol, dt_min,
            options, adjoint_options
        )

    ys = _SdeintAdjointMethod.apply(  # noqa
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

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

"""Define the class of the adjoint SDE when the original forward SDE has additive noise."""

import torch

from .. import base_sde
from .. import misc
from ... import settings


class AdjointSDEAdditive(base_sde.AdjointSDE):

    def __init__(self, sde, params):
        super(AdjointSDEAdditive, self).__init__(base_sde=sde, noise_type=settings.NOISE_TYPES.additive)
        self.params = params

    def f(self, t, y_aug):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            minus_adj_y = [-adj_y_.detach() for adj_y_ in adj_y]
            minus_f = [-f_ for f_ in sde.f(-t, y)]
            vjp_y_and_params = misc.grad(
                outputs=minus_f,
                inputs=y + params,
                grad_outputs=minus_adj_y,
                allow_unused=True,
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

        return (*minus_f, *vjp_y, vjp_params)

    def g_prod(self, t, y_aug, v):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            minus_adj_y = [-adj_y_.detach() for adj_y_ in adj_y]
            minus_g = [-g_ for g_ in sde.g(-t, y)]
            minus_g_prod = misc.seq_batch_mvp(minus_g, v)
            minus_g_weighted = [(minus_g_ * v_.unsqueeze(-2)).sum(-1) for minus_g_, v_ in zip(minus_g, v)]
            vjp_y_and_params = misc.grad(
                outputs=minus_g_weighted,
                inputs=y + params,
                grad_outputs=minus_adj_y,
                allow_unused=True,
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

        return (*minus_g_prod, *vjp_y, vjp_params)

    def g(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")

    def h(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")

    def gdg_prod(self, t, y, v):
        return [0.] * len(y)


class AdjointSDEAdditiveLogqp(base_sde.AdjointSDE):
    def __init__(self, sde, params):
        super(AdjointSDEAdditiveLogqp, self).__init__(base_sde=sde, noise_type=settings.NOISE_TYPES.additive)
        self.params = params

    def f(self, t, y_aug):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 3
        y, adj_y, adj_l = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors], y_aug[2 * n_tensors:3 * n_tensors]
        vjp_l = [torch.zeros_like(adj_l_) for adj_l_ in adj_l]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach() for adj_y_ in adj_y]

            f_eval = sde.f(-t, y)
            f_eval = [-f_eval_ for f_eval_ in f_eval]
            vjp_y_and_params = misc.grad(
                outputs=f_eval,
                inputs=y + params,
                grad_outputs=[-adj_y_ for adj_y_ in adj_y],
                allow_unused=True,
                create_graph=True
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

            # Vector field change due to log-ratio term, i.e. ||u||^2 / 2.
            g_eval = sde.g(-t, y)
            h_eval = sde.h(-t, y)

            g_inv_eval = [torch.pinverse(g_eval_) for g_eval_ in g_eval]
            u_eval = misc.seq_sub(f_eval, h_eval)
            u_eval = [torch.bmm(g_inv_eval_, u_eval_) for g_inv_eval_, u_eval_ in zip(g_inv_eval, u_eval)]
            log_ratio_correction = [.5 * torch.sum(u_eval_ ** 2., dim=1) for u_eval_ in u_eval]
            corr_vjp_y_and_params = misc.grad(
                outputs=log_ratio_correction,
                inputs=y + params,
                grad_outputs=adj_l,
                allow_unused=True,
            )
            corr_vjp_y, corr_vjp_params = corr_vjp_y_and_params[:n_tensors], corr_vjp_y_and_params[n_tensors:]
            corr_vjp_params = misc.flatten(corr_vjp_params)

            vjp_y = misc.seq_add(vjp_y, corr_vjp_y)
            vjp_params = vjp_params + corr_vjp_params

        return (*f_eval, *vjp_y, *vjp_l, vjp_params)

    def g_prod(self, t, y_aug, v):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 3
        y, adj_y, adj_l = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors], y_aug[2 * n_tensors:3 * n_tensors]
        vjp_l = [torch.zeros_like(adj_l_) for adj_l_ in adj_l]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach() for adj_y_ in adj_y]

            g_eval = [-g_ for g_ in sde.g(-t, y)]
            g_prod_eval = misc.seq_batch_mvp(g_eval, v)
            vjp_y_and_params = misc.grad(
                outputs=g_eval,
                inputs=y + params,
                grad_outputs=[-v_.unsqueeze(1) * adj_y_.unsqueeze(2) for v_, adj_y_ in zip(v, adj_y)],
                allow_unused=True,
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

        return (*g_prod_eval, *vjp_y, *vjp_l, vjp_params)

    def g(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")

    def h(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")

    def gdg_prod(self, t, y, v):
        return [0.] * len(y)

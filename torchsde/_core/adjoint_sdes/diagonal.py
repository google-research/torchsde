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

"""Define the class of the adjoint SDE when the original forward SDE has diagonal noise."""
import torch

from .. import base_sde
from .. import misc
from ... import settings


class AdjointSDEDiagonal(base_sde.AdjointSDE):

    def __init__(self, sde, params):
        super(AdjointSDEDiagonal, self).__init__(base_sde=sde, noise_type=settings.NOISE_TYPES.diagonal)
        self.params = params

    def f(self, t, y_aug):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach() for adj_y_ in adj_y]

            g_eval = sde.g(-t, y)
            gdg = misc.grad(
                outputs=g_eval,
                inputs=y,
                grad_outputs=g_eval,
                allow_unused=True,
                create_graph=True
            )
            f_eval = sde.f(-t, y)
            f_eval_corrected = misc.seq_sub(gdg, f_eval)  # Stratonovich correction for reverse-time.
            vjp_y_and_params = misc.grad(
                outputs=f_eval_corrected,
                inputs=y + params,
                grad_outputs=[-adj_y_ for adj_y_ in adj_y],
                allow_unused=True,
                create_graph=True
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

            adj_times_dgdx = misc.grad(
                outputs=g_eval,
                inputs=y,
                grad_outputs=adj_y,
                allow_unused=True,
                create_graph=True
            )

            # This extra term is due to converting the *adjoint* Stratonovich backward SDE to Itô.
            extra_vjp_y_and_params = misc.grad(
                outputs=g_eval,
                inputs=y + params,
                grad_outputs=adj_times_dgdx,
                allow_unused=True,
            )
            extra_vjp_y, extra_vjp_params = extra_vjp_y_and_params[:n_tensors], extra_vjp_y_and_params[n_tensors:]
            extra_vjp_params = misc.flatten(extra_vjp_params)

            vjp_y = misc.seq_add(vjp_y, extra_vjp_y)
            vjp_params = vjp_params + extra_vjp_params

        return (*f_eval_corrected, *vjp_y, vjp_params)

    def g_prod(self, t, y_aug, v):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach() for adj_y_ in adj_y]

            g_eval = [-g_ for g_ in sde.g(-t, y)]
            g_prod_eval = misc.seq_mul(g_eval, v)
            vjp_y_and_params = misc.grad(
                outputs=g_eval,
                inputs=y + params,
                grad_outputs=[-v_ * adj_y_ for v_, adj_y_ in zip(v, adj_y)],
                allow_unused=True,
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

        return (*g_prod_eval, *vjp_y, vjp_params)

    def gdg_prod(self, t, y_aug, v):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach().requires_grad_(True) for adj_y_ in adj_y]

            g_eval = sde.g(-t, y)
            gdg_times_v = misc.grad(
                outputs=g_eval, inputs=y,
                grad_outputs=misc.seq_mul(g_eval, v),
                allow_unused=True,
                create_graph=True,
            )
            dgdy = misc.grad(
                outputs=g_eval, inputs=y,
                grad_outputs=[torch.ones_like(y_) for y_ in y],
                allow_unused=True,
                create_graph=True,
            )
            prod_partials_adj_y_and_params = misc.grad(
                outputs=g_eval, inputs=y + params,
                grad_outputs=misc.seq_mul(adj_y, v, dgdy),
                allow_unused=True,
                create_graph=True,
            )
            prod_partials_adj_y = prod_partials_adj_y_and_params[:n_tensors]
            prod_partials_params = prod_partials_adj_y_and_params[n_tensors:]

            gdg_v = misc.grad(
                outputs=g_eval, inputs=y,
                grad_outputs=[p.detach() for p in misc.seq_mul(adj_y, v, g_eval)],
                allow_unused=True, create_graph=True
            )
            mixed_partials_adj_y_and_params = misc.grad(
                outputs=gdg_v, inputs=y + params,
                grad_outputs=[torch.ones_like(p) for p in gdg_v],
                allow_unused=True,
            )
            mixed_partials_adj_y = mixed_partials_adj_y_and_params[:n_tensors]
            mixed_partials_params = mixed_partials_adj_y_and_params[n_tensors:]
            mixed_partials_params = misc.flatten(mixed_partials_params)

        return (
            *gdg_times_v,
            *misc.seq_sub(prod_partials_adj_y, mixed_partials_adj_y),
            prod_partials_params - mixed_partials_params
        )

    def g(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")

    def h(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")


class AdjointSDEDiagonalLogqp(base_sde.AdjointSDE):

    def __init__(self, sde, params):
        super(AdjointSDEDiagonalLogqp, self).__init__(base_sde=sde, noise_type=settings.NOISE_TYPES.diagonal)
        self.params = params

    def f(self, t, y_aug):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 3
        y, adj_y, adj_l = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors], y_aug[2 * n_tensors:3 * n_tensors]
        vjp_l = [torch.zeros_like(adj_l_) for adj_l_ in adj_l]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach() for adj_y_ in adj_y]

            g_eval = sde.g(-t, y)
            gdg = misc.grad(
                outputs=g_eval,
                inputs=y,
                grad_outputs=g_eval,
                allow_unused=True,
                create_graph=True,
            )

            f_eval = sde.f(-t, y)
            f_eval_corrected = misc.seq_sub(gdg, f_eval)
            vjp_y_and_params = misc.grad(
                outputs=f_eval_corrected, inputs=y + params,
                grad_outputs=[-adj_y_ for adj_y_ in adj_y],
                allow_unused=True,
                create_graph=True
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

            adj_times_dgdx = misc.grad(
                outputs=g_eval, inputs=y,
                grad_outputs=adj_y,
                allow_unused=True,
                create_graph=True
            )
            extra_vjp_y_and_params = misc.grad(
                outputs=g_eval, inputs=y + params,
                grad_outputs=adj_times_dgdx,
                allow_unused=True,
                create_graph=True,
            )
            extra_vjp_y, extra_vjp_params = extra_vjp_y_and_params[:n_tensors], extra_vjp_y_and_params[n_tensors:]
            extra_vjp_params = misc.flatten(extra_vjp_params)

            # Vector field change due to log-ratio term, i.e. ||u||^2 / 2.
            h_eval = sde.h(-t, y)
            u_eval = misc.seq_sub_div(f_eval, h_eval, g_eval)
            log_ratio_correction = [.5 * torch.sum(u_eval_ ** 2., dim=1) for u_eval_ in u_eval]
            corr_vjp_y_and_params = misc.grad(
                outputs=log_ratio_correction, inputs=y + params,
                grad_outputs=adj_l,
                allow_unused=True,
            )
            corr_vjp_y, corr_vjp_params = corr_vjp_y_and_params[:n_tensors], corr_vjp_y_and_params[n_tensors:]
            corr_vjp_params = misc.flatten(corr_vjp_params)

            vjp_y = misc.seq_add(vjp_y, extra_vjp_y, corr_vjp_y)
            vjp_params = vjp_params + extra_vjp_params + corr_vjp_params

        return (*f_eval_corrected, *vjp_y, *vjp_l, vjp_params)

    def g_prod(self, t, y_aug, v):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 3
        y, adj_y, adj_l = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors], y_aug[2 * n_tensors:3 * n_tensors]
        vjp_l = [torch.zeros_like(adj_l_) for adj_l_ in adj_l]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach() for adj_y_ in adj_y]

            g_eval = sde.g(-t, y)
            minus_g_eval = [-g_ for g_ in g_eval]
            minus_g_prod_eval = misc.seq_mul(minus_g_eval, v)

            vjp_y_and_params = misc.grad(
                outputs=minus_g_eval,
                inputs=y + params,
                grad_outputs=[-v_ * adj_y_ for v_, adj_y_ in zip(v, adj_y)],
                allow_unused=True
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

        return (*minus_g_prod_eval, *vjp_y, *vjp_l, vjp_params)

    def gdg_prod(self, t, y_aug, v):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 3
        y, adj_y, adj_l = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors], y_aug[2 * n_tensors:3 * n_tensors]
        vjp_l = [torch.zeros_like(adj_l_) for adj_l_ in adj_l]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            adj_y = [adj_y_.detach().requires_grad_(True) for adj_y_ in adj_y]

            g_eval = sde.g(-t, y)
            gdg_times_v = misc.grad(
                outputs=g_eval,
                inputs=y,
                grad_outputs=misc.seq_mul(g_eval, v),
                allow_unused=True,
                create_graph=True,
            )
            dgdy = misc.grad(
                outputs=g_eval,
                inputs=y,
                grad_outputs=[torch.ones_like(y_) for y_ in y],
                allow_unused=True,
                create_graph=True,
            )

            prod_partials_adj_y_and_params = misc.grad(
                outputs=g_eval,
                inputs=y + params,
                grad_outputs=misc.seq_mul(adj_y, v, dgdy),
                allow_unused=True,
                create_graph=True,
            )
            prod_partials_adj_y = prod_partials_adj_y_and_params[:n_tensors]
            prod_partials_params = prod_partials_adj_y_and_params[n_tensors:]
            prod_partials_params = misc.flatten(prod_partials_params)

            gdg_v = misc.grad(
                outputs=g_eval,
                inputs=y,
                grad_outputs=[p.detach() for p in misc.seq_mul(adj_y, v, g_eval)],
                allow_unused=True,
                create_graph=True,
            )
            gdg_v = [gdg_v_.sum() for gdg_v_ in gdg_v]
            mixed_partials_adj_y_and_params = misc.grad(
                outputs=gdg_v,
                inputs=y + params,
                allow_unused=True,
            )
            mixed_partials_adj_y = mixed_partials_adj_y_and_params[:n_tensors]
            mixed_partials_params = mixed_partials_adj_y_and_params[n_tensors:]
            mixed_partials_params = misc.flatten(mixed_partials_params)

        return (
            *gdg_times_v,
            *misc.seq_sub(prod_partials_adj_y, mixed_partials_adj_y),
            *vjp_l,
            prod_partials_params - mixed_partials_params
        )

    def g(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")

    def h(self, t, y):
        raise NotImplementedError("This method shouldn't be called.")

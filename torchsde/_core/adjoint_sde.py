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

from . import base_sde
from . import misc
from ..settings import SDE_TYPES, NOISE_TYPES


class AdjointSDE(base_sde.BaseSDE):

    def __init__(self, forward_sde, params, logqp=False):
        # There's a mapping from the noise type of the forward SDE to the noise type of the adjoint.
        # Usually, these two aren't the same, e.g. when the forward SDE has additive noise, the adjoint SDE's diffusion
        # is a linear function of the adjoint variable, so it is not of additive noise.
        sde_type = forward_sde.sde_type
        if sde_type == SDE_TYPES.ito:
            noise_type = {
                NOISE_TYPES.diagonal: NOISE_TYPES.diagonal,
                NOISE_TYPES.additive: NOISE_TYPES.general,
                NOISE_TYPES.scalar: NOISE_TYPES.scalar,
                NOISE_TYPES.general: NOISE_TYPES.general,
            }[forward_sde.noise_type]
        else:
            noise_type = {
                NOISE_TYPES.general: NOISE_TYPES.general,
                NOISE_TYPES.additive: NOISE_TYPES.general,
                NOISE_TYPES.scalar: NOISE_TYPES.scalar,
                NOISE_TYPES.diagonal: NOISE_TYPES.diagonal,
            }[forward_sde.noise_type]

        super(AdjointSDE, self).__init__(sde_type=sde_type, noise_type=noise_type)
        self._base_sde = forward_sde
        self._params = params

        # Register the core function. This avoids polluting the codebase with if-statements and speeds things up.
        if logqp:
            self.f = {
                SDE_TYPES.ito: {
                    NOISE_TYPES.diagonal: self.f_corrected_diagonal_logqp,
                }.get(noise_type, self.f_corrected_default_logqp),
                SDE_TYPES.stratonovich: self.f_uncorrected_logqp
            }[sde_type]
        else:
            self.f = {
                SDE_TYPES.ito: {
                    NOISE_TYPES.diagonal: self.f_corrected_diagonal,
                }.get(noise_type, self.f_corrected_default),
                SDE_TYPES.stratonovich: self.f_uncorrected
            }[sde_type]

        self.g_prod = {
            NOISE_TYPES.diagonal: self.g_prod_diagonal
        }.get(noise_type, self.g_prod_default)

        self.gdg_prod = {
            NOISE_TYPES.diagonal: self.gdg_prod_diagonal,
            NOISE_TYPES.additive: self._skip
        }.get(noise_type, self.gdg_prod_default)

    # f functions.
    def f_uncorrected(self, t, y_aug):  # For Stratonovich.
        sde, params, n_tensors = self._base_sde, self._params, len(y_aug) // 2
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

    def f_corrected_default(self, t, y_aug):  # For Ito general/scalar.
        # TODO: This requires 2 corrections: One in the forward, and the other in backward.
        raise NotImplementedError

    def f_corrected_diagonal(self, t, y_aug):  # For Ito diagonal.
        sde, params, n_tensors = self._base_sde, self._params, len(y_aug) // 2
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
            # Stratonovich correction for reverse-time.
            f_eval_corrected = misc.seq_sub(gdg, f_eval)
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

            # Converting the *adjoint* Stratonovich backward SDE to Itô.
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

    # g_prod functions.
    def g_prod_default(self, t, y_aug, v):  # For Ito/Stratonovich general/additive/scalar.
        sde, params, n_tensors = self._base_sde, self._params, len(y_aug) // 2
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

    def g_prod_diagonal(self, t, y_aug, v):  # For Ito/Stratonovich diagonal.
        sde, params, n_tensors = self._base_sde, self._params, len(y_aug) // 2
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

    # gdg_prod functions.
    def gdg_prod_diagonal(self, t, y_aug, v):  # For Ito/Stratonovich diagonal.
        sde, params, n_tensors = self._base_sde, self._params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

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
                create_graph=True
            )
            mixed_partials_adj_y_and_params = misc.grad(
                outputs=gdg_v,
                inputs=y + params,
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

    def gdg_prod_default(self, t, y, v):  # For Ito/Stratonovich general/additive/scalar.
        # TODO: Write this!
        raise NotImplementedError

    def f_uncorrected_logqp(self, t, y_aug):
        sde, params, n_tensors = self._base_sde, self._params, len(y_aug) // 3
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

    def f_corrected_default_logqp(self, t, y_aug):
        # TODO: This requires 2 corrections: One in the forward, and the other in backward.
        raise NotImplementedError

    def f_corrected_diagonal_logqp(self, t, y_aug):
        sde, params, n_tensors = self._base_sde, self._params, len(y_aug) // 3
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

    def _skip(self, *args):  # noqa
        _, y = args[:2]
        return [0.] * len(y)

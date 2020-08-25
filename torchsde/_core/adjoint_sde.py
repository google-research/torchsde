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


from typing import Sequence

import torch

from . import base_sde
from . import misc
from ..settings import SDE_TYPES, NOISE_TYPES
from ..types import TensorOrTensors


class AdjointSDE(base_sde.BaseSDE):

    def __init__(self,
                 sde: base_sde.ForwardSDE,
                 params: TensorOrTensors,
                 shapes: Sequence[torch.Size]):
        # There's a mapping from the noise type of the forward SDE to the noise type of the adjoint.
        # Usually, these two aren't the same, e.g. when the forward SDE has additive noise, the adjoint SDE's diffusion
        # is a linear function of the adjoint variable, so it is not of additive noise.
        sde_type = sde.sde_type
        noise_type = {
            NOISE_TYPES.general: NOISE_TYPES.general,
            NOISE_TYPES.additive: NOISE_TYPES.general,
            NOISE_TYPES.scalar: NOISE_TYPES.scalar,
            NOISE_TYPES.diagonal: NOISE_TYPES.diagonal,
        }.get(sde.noise_type)
        super(AdjointSDE, self).__init__(sde_type=sde_type, noise_type=noise_type)

        self._base_sde = sde
        self._params = params
        self._shapes = shapes

        # Register the core functions. This avoids polluting the codebase with if-statements and achieves speed-ups
        # by making sure it's a one-time cost. The `sde_type` and `noise_type` of the forward SDE determines the
        # registered functions.
        self.f = {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: self.f_corrected_diagonal,
                NOISE_TYPES.additive: self.f_uncorrected,
                NOISE_TYPES.scalar: self.f_corrected_default,
                NOISE_TYPES.general: self.f_corrected_default
            }.get(sde.noise_type),
            SDE_TYPES.stratonovich: self.f_uncorrected
        }.get(sde.sde_type)
        self.gdg_prod = {
            NOISE_TYPES.diagonal: self.gdg_prod_diagonal,
        }.get(sde.noise_type, self.gdg_prod_default)

    def _unpack_y_aug(self, y_aug):
        y, adj_y = misc.flat_to_shape(y_aug, self._shapes[:2])
        return y.detach().requires_grad_(True), adj_y.detach().requires_grad_(True)

    def _flat_to_shape(self, y_aug):
        """Recover only first two tensors from the flattened augmented state."""
        return misc.flat_to_shape(y_aug, self._shapes[:2])

    ########################################
    #                  f                   #
    ########################################

    def f_uncorrected(self, t, y_aug):  # For Ito additive and Stratonovich.
        with torch.enable_grad():
            y, adj_y = self._unpack_y_aug(y_aug)
            f = self._base_sde.f(-t, y)
            vjp_y_and_params = misc.grad(
                outputs=f,
                inputs=[y] + self._params,
                grad_outputs=adj_y,
                allow_unused=True
            )
        return misc.flatten((-f, *vjp_y_and_params))

    def f_corrected_default(self, t, y_aug):  # For Ito general/scalar.
        raise NotImplementedError

    def f_corrected_diagonal(self, t, y_aug):  # For Ito diagonal.
        with torch.enable_grad():
            y, adj_y = self._unpack_y_aug(y_aug)
            g = self._base_sde.g(-t, y)
            g_dg_vjp, = misc.grad(
                outputs=g,
                inputs=y,
                grad_outputs=g,
                allow_unused=True,
                create_graph=True
            )
            # Double Stratonovich correction.
            f = self._base_sde.f(-t, y) - g_dg_vjp
            vjp_y_and_params = misc.grad(
                outputs=f,
                inputs=[y] + self._params,
                grad_outputs=adj_y,
                allow_unused=True,
                create_graph=True
            )
            # Convert the adjoint Stratonovich SDE to Itô form.
            a_dg_vjp, = misc.grad(
                outputs=g,
                inputs=y,
                grad_outputs=adj_y,
                allow_unused=True,
                create_graph=True
            )
            extra_vjp_y_and_params = misc.grad(
                outputs=g,
                inputs=[y] + self._params,
                grad_outputs=a_dg_vjp,
                allow_unused=True,
            )
            vjp_y_and_params = misc.seq_add(vjp_y_and_params, extra_vjp_y_and_params)
        return misc.flatten((-f, *vjp_y_and_params))

    ########################################
    #                  g                   #
    ########################################

    def g(self, t, y):
        # We don't want to define it, it's super inefficient to compute.
        # In theory every part of the code which _could_ call it either does something else, or has some more
        # informative error message to tell the user what went wrong.
        # This is here as a fallback option.
        raise RuntimeError("Adjoint `g` not defined. Please report a bug to torchsde.")

    ########################################
    #                g_prod                #
    ########################################

    def g_prod(self, t, y_aug, v):
        with torch.enable_grad():
            y, adj_y = self._unpack_y_aug(y_aug)
            g_prod = self._base_sde.g_prod(-t, y, v)
            vjp_y_and_params = misc.grad(
                outputs=g_prod,
                inputs=[y] + self._params,
                grad_outputs=adj_y,
                allow_unused=True,
            )
        return misc.flatten((-g_prod, *vjp_y_and_params))

    ########################################
    #               gdg_prod               #
    ########################################

    def gdg_prod_default(self, t, y, v):  # For Ito/Stratonovich general/additive/scalar.
        raise NotImplementedError

    def gdg_prod_diagonal(self, t, y_aug, v):  # For Ito/Stratonovich diagonal.
        with torch.enable_grad():
            y, adj_y = self._unpack_y_aug(y_aug)
            g = self._base_sde.g(-t, y)
            vg_dg_vjp, = misc.grad(
                outputs=g,
                inputs=y,
                grad_outputs=v * g,
                allow_unused=True,
                create_graph=True,
            )
            dgdy, = misc.grad(
                outputs=g.sum(),
                inputs=y,
                allow_unused=True,
                create_graph=True,
            )
            prod_partials_adj_y_and_params = misc.grad(
                outputs=g,
                inputs=[y] + self._params,
                grad_outputs=adj_y * v * dgdy,
                allow_unused=True,
                create_graph=True
            )
            avg_dg_vjp, = misc.grad(
                outputs=g,
                inputs=y,
                grad_outputs=(adj_y * v * g).detach(),
                allow_unused=True,
                create_graph=True
            )
            mixed_partials_adj_y_and_params = misc.grad(
                outputs=avg_dg_vjp.sum(),
                inputs=[y] + self._params,
                allow_unused=True,
            )
            vjp_y_and_params = misc.seq_sub(prod_partials_adj_y_and_params, mixed_partials_adj_y_and_params)
        return misc.flatten((vg_dg_vjp, *vjp_y_and_params))

    ########################################
    #              dg_ga_jvp               #
    ########################################

    # Diffusion of adjoint outputs matrix of size (d + d + p) x m,
    # where d is the dimension of state, p is the dimension of parameter,
    # and m is the dimension of Brownian motion.

    # Indexing the diffusion of the original system by column, the diffusion of the adjoint is
    #     -g_1                -g_2          ...         -g_m
    # dg_1 dy^T a         dg_2 dy^T a       ...     dg_m dy^T a
    # dg_1 dθ^T a         dg_2 dθ^T a       ...     dg_m dθ^T a

    def dg_ga_jvp_column_sum_v1(self, t, y_aug, a):
        # Assumes `a` is anti-symmetric and `_base_sde` is not of diagonal noise.
        with torch.enable_grad():
            y, adj_y = self._unpack_y_aug(y_aug)
            g = self._base_sde.g(-t, y)

            adj_g = [
                misc.grad(outputs=g[..., col_idx],
                          inputs=[y] + self._params,
                          grad_outputs=adj_y,
                          create_graph=True,
                          allow_unused=True)
                for col_idx in range(g.size(-1))
            ]
            # Transpose nested list and stack columns.
            adj_g = [-g] + [torch.stack(args, dim=-1) for args in zip(*adj_g)]
            # Matmul with Levy area. The term for parameters isn't exactly correct,
            # since it's approximating \sum_i a_i b_i with (\sum_i a_i) (\sum_i b_i).
            # The problem is due to there not being a simple way to consider parameter
            # gradients for each batch entry individually.
            adj_ga = [torch.bmm(t.detach(), a) for t in adj_g[:2]]
            adj_ga += [torch.mm(t.detach().flatten(0, -1).unsqueeze(0), a.sum(0)).reshape(t.size()) for t in adj_g[2:]]

            # Need the unflattened structure here, since inputs must be unflattened.
            dg_ga_jvp = [
                misc.jvp(
                    outputs=[t[..., col_idx] for t in adj_g],
                    inputs=[y, adj_y] + self._params,
                    grad_inputs=[t[..., col_idx] for t in adj_ga],
                    retain_graph=True,
                    allow_unused=True
                )
                for col_idx in range(g.size(-1))
            ]
            dg_ga_jvp = [sum(args) for args in zip(*dg_ga_jvp)]
        return misc.flatten(dg_ga_jvp)

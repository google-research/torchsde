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
from ..settings import NOISE_TYPES, SDE_TYPES
from ..types import Sequence, TensorOrTensors


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
        self.f_and_g_prod = {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: self.f_and_g_prod_corrected_diagonal,
                NOISE_TYPES.additive: self.f_and_g_prod_uncorrected,
                NOISE_TYPES.scalar: self.f_and_g_prod_corrected_default,
                NOISE_TYPES.general: self.f_and_g_prod_corrected_default
            }.get(sde.noise_type),
            SDE_TYPES.stratonovich: self.f_and_g_prod_uncorrected
        }.get(sde.sde_type)
        self.g_prod_and_gdg_prod = {
            NOISE_TYPES.diagonal: self.g_prod_and_gdg_prod_diagonal,
        }.get(sde.noise_type, self.g_prod_and_gdg_prod_default)

    ########################################
    #            Helper functions          #
    ########################################

    def _get_state(self, t, y_aug, v=None):
        """Unpacks y_aug, whilst enforcing the necessary checks so that we can calculate derivatives wrt state."""

        # These leaf checks are very important.
        # _get_state is used where we want to compute:
        # ```
        # with torch.enable_grad():
        #     s = some_function(y)
        #     torch.autograd.grad(s, [y] + params, ...)
        # ```
        # where `some_function` implicitly depends on `params`.
        # However if y has history of its own then in principle it could _also_ depend upon `params`, and this call to
        # `grad` will go all the way back to that. To avoid this, we require that every input tensor be a leaf tensor.
        #
        # This is also the reason for the `y0.detach()` in adjoint.py::_SdeintAdjointMethod.forward. If we don't detach,
        # then y0 may have a history and these checks will fail. This is a spurious failure as
        # `torch.autograd.Function.forward` has an implicit `torch.no_grad()` guard, i.e. we definitely don't want to
        # use its history there.
        assert t.is_leaf, "Internal error: please report a bug to torchsde"
        assert y_aug.is_leaf, "Internal error: please report a bug to torchsde"
        if v is not None:
            assert v.is_leaf, "Internal error: please report a bug to torchsde"

        requires_grad = torch.is_grad_enabled()

        shapes = self._shapes[:2]
        numel = sum(shape.numel() for shape in shapes)
        y, adj_y = misc.flat_to_shape(y_aug[:numel], shapes)

        # To support the later differentiation wrt y, we set it to require_grad if it doesn't already.
        if not y.requires_grad:
            y = y.detach().requires_grad_(True)
        return y, adj_y, requires_grad

    def _f_uncorrected(self, f, y, adj_y, requires_grad):
        vjp_y_and_params = misc.vjp(
            outputs=f,
            inputs=[y] + self._params,
            grad_outputs=adj_y,
            allow_unused=True,
            retain_graph=True,
            create_graph=requires_grad
        )
        if not requires_grad:
            # We had to build a computational graph to be able to compute the above vjp.
            # However, if we don't require_grad then we don't need to backprop through this function, so we should
            # delete the computational graph to avoid a memory leak. (Which for example would keep the local
            # variable `y` in memory: f->grad_fn->...->AccumulatedGrad->y.)

            # Note: `requires_grad` might not be equal to what `torch.is_grad_enabled` returns here.
            f = f.detach()
        return misc.flatten((-f, *vjp_y_and_params))

    def _f_corrected_default(self, f, g, y, adj_y, requires_grad):
        g_columns = [g_column.squeeze(dim=-1) for g_column in g.split(1, dim=-1)]
        dg_g_jvp = sum([
            misc.jvp(
                outputs=g_column,
                inputs=y,
                grad_inputs=g_column,
                allow_unused=True,
                create_graph=True
            )[0] for g_column in g_columns
        ])
        # Double Stratonovich correction.
        f = f - dg_g_jvp
        vjp_y_and_params = misc.vjp(
            outputs=f,
            inputs=[y] + self._params,
            grad_outputs=adj_y,
            allow_unused=True,
            retain_graph=True,
            create_graph=requires_grad
        )
        # Convert the adjoint Stratonovich SDE to Itô form.
        extra_vjp_y_and_params = []
        for g_column in g_columns:
            a_dg_vjp, = misc.vjp(
                outputs=g_column,
                inputs=y,
                grad_outputs=adj_y,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            extra_vjp_y_and_params_column = misc.vjp(
                outputs=g_column,
                inputs=[y] + self._params,
                grad_outputs=a_dg_vjp,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            extra_vjp_y_and_params.append(extra_vjp_y_and_params_column)
        vjp_y_and_params = misc.seq_add(vjp_y_and_params, *extra_vjp_y_and_params)
        if not requires_grad:
            # See corresponding note in _f_uncorrected.
            f = f.detach()
        return misc.flatten((-f, *vjp_y_and_params))

    def _f_corrected_diagonal(self, f, g, y, adj_y, requires_grad):
        g_dg_vjp, = misc.vjp(
            outputs=g,
            inputs=y,
            grad_outputs=g,
            allow_unused=True,
            create_graph=True
        )
        # Double Stratonovich correction.
        f = f - g_dg_vjp
        vjp_y_and_params = misc.vjp(
            outputs=f,
            inputs=[y] + self._params,
            grad_outputs=adj_y,
            allow_unused=True,
            retain_graph=True,
            create_graph=requires_grad
        )
        # Convert the adjoint Stratonovich SDE to Itô form.
        a_dg_vjp, = misc.vjp(
            outputs=g,
            inputs=y,
            grad_outputs=adj_y,
            allow_unused=True,
            retain_graph=True,
            create_graph=requires_grad
        )
        extra_vjp_y_and_params = misc.vjp(
            outputs=g,
            inputs=[y] + self._params,
            grad_outputs=a_dg_vjp,
            allow_unused=True,
            retain_graph=True,
            create_graph=requires_grad
        )
        vjp_y_and_params = misc.seq_add(vjp_y_and_params, extra_vjp_y_and_params)
        if not requires_grad:
            # See corresponding note in _f_uncorrected.
            f = f.detach()
        return misc.flatten((-f, *vjp_y_and_params))

    def _g_prod(self, g_prod, y, adj_y, requires_grad):
        vjp_y_and_params = misc.vjp(
            outputs=g_prod,
            inputs=[y] + self._params,
            grad_outputs=adj_y,
            allow_unused=True,
            retain_graph=True,
            create_graph=requires_grad
        )
        if not requires_grad:
            # See corresponding note in _f_uncorrected.
            g_prod = g_prod.detach()
        return misc.flatten((-g_prod, *vjp_y_and_params))

    ########################################
    #                  f                   #
    ########################################

    def f_uncorrected(self, t, y_aug):  # For Ito additive and Stratonovich.
        y, adj_y, requires_grad = self._get_state(t, y_aug)
        with torch.enable_grad():
            f = self._base_sde.f(-t, y)
            return self._f_uncorrected(f, y, adj_y, requires_grad)

    def f_corrected_default(self, t, y_aug):  # For Ito general/scalar.
        y, adj_y, requires_grad = self._get_state(t, y_aug)
        with torch.enable_grad():
            f, g = self._base_sde.f_and_g(-t, y)
            return self._f_corrected_default(f, g, y, adj_y, requires_grad)

    def f_corrected_diagonal(self, t, y_aug):  # For Ito diagonal.
        y, adj_y, requires_grad = self._get_state(t, y_aug)
        with torch.enable_grad():
            f, g = self._base_sde.f_and_g(-t, y)
            return self._f_corrected_diagonal(f, g, y, adj_y, requires_grad)

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
    #               f_and_g                #
    ########################################

    def f_and_g(self, t, y):
        # Like g above, this is inefficient to compute.
        raise RuntimeError("Adjoint `f_and_g` not defined. Please report a bug to torchsde.")

    ########################################
    #                prod                  #
    ########################################

    def prod(self, g, v):
        # We could define this just fine, but we don't expect to ever be able to compute the input `g`, so we should
        # never get here.
        raise RuntimeError("Adjoint `prod` not defined. Please report a bug to torchsde.")

    ########################################
    #                g_prod                #
    ########################################

    def g_prod(self, t, y_aug, v):
        y, adj_y, requires_grad = self._get_state(t, y_aug, v)
        with torch.enable_grad():
            g_prod = self._base_sde.g_prod(-t, y, v)
            return self._g_prod(g_prod, y, adj_y, requires_grad)

    ########################################
    #            f_and_g_prod              #
    ########################################

    def f_and_g_prod_uncorrected(self, t, y_aug, v):  # For Ito additive and Stratonovich.
        y, adj_y, requires_grad = self._get_state(t, y_aug)
        with torch.enable_grad():
            f, g_prod = self._base_sde.f_and_g_prod(-t, y, v)

            f_out = self._f_uncorrected(f, y, adj_y, requires_grad)
            g_prod_out = self._g_prod(g_prod, y, adj_y, requires_grad)
            return f_out, g_prod_out

    def f_and_g_prod_corrected_default(self, t, y_aug, v):  # For Ito general/scalar.
        y, adj_y, requires_grad = self._get_state(t, y_aug)
        with torch.enable_grad():
            f, g = self._base_sde.f_and_g(-t, y)
            g_prod = self._base_sde.prod(g, v)

            f_out = self._f_corrected_default(f, g, y, adj_y, requires_grad)
            g_prod_out = self._g_prod(g_prod, y, adj_y, requires_grad)
            return f_out, g_prod_out

    def f_and_g_prod_corrected_diagonal(self, t, y_aug, v):  # For Ito diagonal.
        y, adj_y, requires_grad = self._get_state(t, y_aug)
        with torch.enable_grad():
            f, g = self._base_sde.f_and_g(-t, y)
            g_prod = self._base_sde.prod(g, v)

            f_out = self._f_corrected_diagonal(f, g, y, adj_y, requires_grad)
            g_prod_out = self._g_prod(g_prod, y, adj_y, requires_grad)
            return f_out, g_prod_out

    ########################################
    #               gdg_prod               #
    ########################################

    def g_prod_and_gdg_prod_default(self, t, y, v1, v2):  # For Ito/Stratonovich general/additive/scalar.
        raise NotImplementedError

    def g_prod_and_gdg_prod_diagonal(self, t, y_aug, v1, v2):  # For Ito/Stratonovich diagonal.
        y, adj_y, requires_grad = self._get_state(t, y_aug, v2)
        with torch.enable_grad():
            g = self._base_sde.g(-t, y)
            g_prod = self._base_sde.prod(g, v1)

            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=v2 * g,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            dgdy, = misc.vjp(
                outputs=g.sum(),
                inputs=y,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            prod_partials_adj_y_and_params = misc.vjp(
                outputs=g,
                inputs=[y] + self._params,
                grad_outputs=adj_y * v2 * dgdy,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            avg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=(adj_y * v2 * g).detach(),
                allow_unused=True,
                create_graph=True
            )
            mixed_partials_adj_y_and_params = misc.vjp(
                outputs=avg_dg_vjp.sum(),
                inputs=[y] + self._params,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            vjp_y_and_params = misc.seq_sub(prod_partials_adj_y_and_params, mixed_partials_adj_y_and_params)
            return self._g_prod(g_prod, y, adj_y, requires_grad), misc.flatten((vg_dg_vjp, *vjp_y_and_params))

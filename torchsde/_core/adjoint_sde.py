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
from ..types import TensorOrTensors, Sequence


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

    def _unpack_y_aug(self, t, y_aug, v):
        # These leaf checks are very important.
        # _unpack_y_aug is used where we want to compute:
        # ```
        # with torch.enable_grad():
        #     s = some_function(y)
        #     torch.autograd.grad(s, [y] + params, ...)
        # ```
        # where `some_function` implicitly depends on `params`.
        # However if y has history of its own then in principle it could _also_ depend upon `params`, and this call to
        # `grad` will go all the way back to that.
        # To avoid this, we require that y (and the other tensors) be leaf tensors.
        #
        # Note that this is also the reason for the `y0.detach()` in adjoint.py::_SdeintAdjointMethod.forward. Unless
        # we detach then y0 may have a history and these checks will fail. Note that this isn't then producing the wrong
        # gradients when backprop'ing as usual because `torch.autograd.Function.forward` has an implicit
        # `torch.no_grad()` guard, i.e. any history doesn't affect the usual gradient computations anyway. But it does
        # still get detected if we call `torch.autograd.grad` ourselves, which is explicitly what we don't want as per
        # the previous paragraph.
        assert t.is_leaf, "Internal error: please report a bug to torchsde"
        assert y_aug.is_leaf, "Internal error: please report a bug to torchsde"
        assert v.is_leaf, "Internal error: please report a bug to torchsde"

        # This determines whether or not we will be able to backpropagate through the calling function (that is calling
        # _unpack_y_aug). Simply, we should be able to if and only if (a) gradients are enabled and (b) the input
        # arguments require gradient.
        # Note that we don't fix this to True, because that implies building computational graphs, which will consume
        # additional memory that will be unnecessary if we don't need to backpropagate.
        requires_grad = torch.is_grad_enabled() and (t.requires_grad or y_aug.requires_grad or v.requires_grad)

        y, adj_y = misc.flat_to_shape(y_aug, self._shapes[:2])

        # To support the later differentiation wrt y, we set it to require_grad if it doesn't already.
        if not y.requires_grad:
            y = y.detach().requires_grad_(True)
        return y, adj_y, requires_grad

    ########################################
    #                  f                   #
    ########################################

    def f_uncorrected(self, t, y_aug):  # For Ito additive and Stratonovich.
        y, adj_y, requires_grad = self._unpack_y_aug(t, y_aug, v=t)  # just use t as a dummy `v`
        with torch.enable_grad():
            f = self._base_sde.f(-t, y)
            vjp_y_and_params = misc.vjp(
                outputs=f,
                inputs=[y] + self._params,
                grad_outputs=adj_y,
                allow_unused=True,
                create_graph=requires_grad
            )
            if not requires_grad:
                # We had to build a computational graph to be able to compute the above vjp.
                # However, if we don't require_grad then we don't need to backprop through this function, so we should
                # delete the computational graph to avoid a memory leak. (Which for example would keep the local
                # variable `y` in memory: f->grad_fn->...->AccumulatedGrad->y.)
                f = f.detach()
        return misc.flatten((-f, *vjp_y_and_params))

    def f_corrected_default(self, t, y_aug):  # For Ito general/scalar.
        raise NotImplementedError

    def f_corrected_diagonal(self, t, y_aug):  # For Ito diagonal.
        y, adj_y, requires_grad = self._unpack_y_aug(t, y_aug, v=t)  # just use t as a dummy `v`
        with torch.enable_grad():
            g = self._base_sde.g(-t, y)
            g_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g,
                allow_unused=True,
                create_graph=True
            )
            # Double Stratonovich correction.
            f = self._base_sde.f(-t, y) - g_dg_vjp
            vjp_y_and_params = misc.vjp(
                outputs=f,
                inputs=[y] + self._params,
                grad_outputs=adj_y,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            if not requires_grad:
                # See corresponding note in f_uncorrected.
                f = f.detach()
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
                create_graph=requires_grad
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
        y, adj_y, requires_grad = self._unpack_y_aug(t, y_aug, v)
        with torch.enable_grad():
            g_prod = self._base_sde.g_prod(-t, y, v)
            vjp_y_and_params = misc.vjp(
                outputs=g_prod,
                inputs=[y] + self._params,
                grad_outputs=adj_y,
                allow_unused=True,
                create_graph=requires_grad
            )
            if not requires_grad:
                # See corresponding note in f_uncorrected.
                g_prod = g_prod.detach()
        return misc.flatten((-g_prod, *vjp_y_and_params))

    ########################################
    #               gdg_prod               #
    ########################################

    def gdg_prod_default(self, t, y, v):  # For Ito/Stratonovich general/additive/scalar.
        raise NotImplementedError

    def gdg_prod_diagonal(self, t, y_aug, v):  # For Ito/Stratonovich diagonal.
        y, adj_y, requires_grad = self._unpack_y_aug(t, y_aug, v)
        with torch.enable_grad():
            g = self._base_sde.g(-t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=v * g,
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
                grad_outputs=adj_y * v * dgdy,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            avg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=(adj_y * v * g).detach(),
                allow_unused=True,
                create_graph=True
            )
            mixed_partials_adj_y_and_params = misc.vjp(
                outputs=avg_dg_vjp.sum(),
                inputs=[y] + self._params,
                allow_unused=True,
                create_graph=requires_grad
            )
            vjp_y_and_params = misc.seq_sub(prod_partials_adj_y_and_params, mixed_partials_adj_y_and_params)
        return misc.flatten((vg_dg_vjp, *vjp_y_and_params))

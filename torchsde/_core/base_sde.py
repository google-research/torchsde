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

import abc

import torch
from torch import nn

from . import misc
from ..settings import NOISE_TYPES, SDE_TYPES


class BaseSDE(abc.ABC, nn.Module):
    """Base class for all SDEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(self, noise_type, sde_type):
        super(BaseSDE, self).__init__()
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"Expected noise type in {NOISE_TYPES}, but found {noise_type}")
        if sde_type not in SDE_TYPES:
            raise ValueError(f"Expected sde type in {SDE_TYPES}, but found {sde_type}")
        # Making these Python properties breaks `torch.jit.script`.
        self.noise_type = noise_type
        self.sde_type = sde_type


class ForwardSDE(BaseSDE):

    def __init__(self, sde, fast_dg_ga_jvp_column_sum=False):
        super(ForwardSDE, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        self._base_sde = sde
        self.f = sde.f
        self.g = sde.g

        # Register the core functions. This avoids polluting the codebase with if-statements and achieves speed-ups
        # by making sure it's a one-time cost.
        self.g_prod = {
            NOISE_TYPES.diagonal: self.g_prod_diagonal,
        }.get(sde.noise_type, self.g_prod_default)
        self.gdg_prod = {
            NOISE_TYPES.diagonal: self.gdg_prod_diagonal,
            NOISE_TYPES.additive: self._return_zero,
        }.get(sde.noise_type, self.gdg_prod_default)
        self.dg_ga_jvp_column_sum = {
            NOISE_TYPES.general: (
                self.dg_ga_jvp_column_sum_v2 if fast_dg_ga_jvp_column_sum else self.dg_ga_jvp_column_sum_v1
            )
        }.get(sde.noise_type, self._return_zero)

    ########################################
    #                g_prod                #
    ########################################

    def g_prod_diagonal(self, t, y, v):
        return self.g(t, y) * v

    def g_prod_default(self, t, y, v):
        return misc.batch_mvp(self.g(t, y), v)

    ########################################
    #               gdg_prod               #
    ########################################

    # Computes: sum_{j, l} g_{j, l} d g_{j, l} d x_i v_l.
    def gdg_prod_default(self, t, y, v):
        requires_grad = torch.is_grad_enabled() and (t.requires_grad or y.requires_grad or v.requires_grad)
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v.unsqueeze(-2),
                create_graph=requires_grad,
                allow_unused=True
            )
        return vg_dg_vjp

    def gdg_prod_diagonal(self, t, y, v):
        requires_grad = torch.is_grad_enabled() and (t.requires_grad or y.requires_grad or v.requires_grad)
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v,
                create_graph=requires_grad,
                allow_unused=True
            )
        return vg_dg_vjp

    ########################################
    #              dg_ga_jvp               #
    ########################################

    # Computes: sum_{j,k,l} d g_{i,l} / d x_j g_{j,k} A_{k,l}.
    def dg_ga_jvp_column_sum_v1(self, t, y, a):
        requires_grad = torch.is_grad_enabled() and (t.requires_grad or y.requires_grad or a.requires_grad)
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)
            dg_ga_jvp = [
                misc.jvp(
                    outputs=g[..., col_idx],
                    inputs=y,
                    grad_inputs=ga[..., col_idx],
                    retain_graph=True,
                    create_graph=requires_grad,
                    allow_unused=True
                )[0]
                for col_idx in range(g.size(-1))
            ]
            dg_ga_jvp = sum(dg_ga_jvp)
        return dg_ga_jvp

    def dg_ga_jvp_column_sum_v2(self, t, y, a):
        # Faster, but more memory intensive.
        requires_grad = torch.is_grad_enabled() and (t.requires_grad or y.requires_grad or a.requires_grad)
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)

            batch_size, d, m = g.size()
            y_dup = torch.repeat_interleave(y, repeats=m, dim=0)
            g_dup = self.g(t, y_dup)
            ga_flat = ga.transpose(1, 2).flatten(0, 1)
            dg_ga_jvp, = misc.jvp(
                outputs=g_dup,
                inputs=y_dup,
                grad_inputs=ga_flat,
                create_graph=requires_grad,
                allow_unused=True
            )
            dg_ga_jvp = dg_ga_jvp.reshape(batch_size, m, d, m).permute(0, 2, 1, 3)
            dg_ga_jvp = dg_ga_jvp.diagonal(dim1=-2, dim2=-1).sum(-1)
        return dg_ga_jvp

    def _return_zero(self, t, y, v):  # noqa
        return 0.


class RenameMethodsSDE(BaseSDE):

    def __init__(self, sde, drift='f', diffusion='g'):
        super(RenameMethodsSDE, self).__init__(noise_type=sde.noise_type, sde_type=sde.sde_type)
        self._base_sde = sde
        self.f = getattr(sde, drift)
        self.g = getattr(sde, diffusion)


class SDEIto(BaseSDE):

    def __init__(self, noise_type):
        super(SDEIto, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.ito)


class SDEStratonovich(BaseSDE):

    def __init__(self, noise_type):
        super(SDEStratonovich, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.stratonovich)

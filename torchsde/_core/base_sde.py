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
from ..types import Tensor


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

        # Register the core functions. This avoids polluting the codebase with if-statements and achieves speed-ups
        # by making sure it's a one-time cost.

        if hasattr(sde, 'f_and_g_prod'):
            self.f_and_g_prod = sde.f_and_g_prod
        elif hasattr(sde, 'f') and hasattr(sde, 'g_prod'):
            self.f_and_g_prod = self.f_and_g_prod_default1
        else:  # (f_and_g,) or (f, g,).
            self.f_and_g_prod = self.f_and_g_prod_default2

        self.f = getattr(sde, 'f', self.f_default)
        self.g = getattr(sde, 'g', self.g_default)
        self.f_and_g = getattr(sde, 'f_and_g', self.f_and_g_default)
        self.g_prod = getattr(sde, 'g_prod', self.g_prod_default)
        self.prod = {
            NOISE_TYPES.diagonal: self.prod_diagonal
        }.get(sde.noise_type, self.prod_default)
        self.g_prod_and_gdg_prod = {
            NOISE_TYPES.diagonal: self.g_prod_and_gdg_prod_diagonal,
            NOISE_TYPES.additive: self.g_prod_and_gdg_prod_additive,
        }.get(sde.noise_type, self.g_prod_and_gdg_prod_default)
        self.dg_ga_jvp_column_sum = {
            NOISE_TYPES.general: (
                self.dg_ga_jvp_column_sum_v2 if fast_dg_ga_jvp_column_sum else self.dg_ga_jvp_column_sum_v1
            )
        }.get(sde.noise_type, self._return_zero)

    ########################################
    #                  f                   #
    ########################################
    def f_default(self, t, y):
        raise RuntimeError("Method `f` has not been provided, but is required for this method.")

    ########################################
    #                  g                   #
    ########################################
    def g_default(self, t, y):
        raise RuntimeError("Method `g` has not been provided, but is required for this method.")

    ########################################
    #               f_and_g                #
    ########################################

    def f_and_g_default(self, t, y):
        return self.f(t, y), self.g(t, y)

    ########################################
    #                prod                  #
    ########################################

    def prod_diagonal(self, g, v):
        return g * v

    def prod_default(self, g, v):
        return misc.batch_mvp(g, v)

    ########################################
    #                g_prod                #
    ########################################

    def g_prod_default(self, t, y, v):
        return self.prod(self.g(t, y), v)

    ########################################
    #             f_and_g_prod             #
    ########################################

    def f_and_g_prod_default1(self, t, y, v):
        return self.f(t, y), self.g_prod(t, y, v)

    def f_and_g_prod_default2(self, t, y, v):
        f, g = self.f_and_g(t, y)
        return f, self.prod(g, v)

    ########################################
    #          g_prod_and_gdg_prod         #
    ########################################

    # Computes: g_prod and sum_{j, l} g_{j, l} d g_{j, l} d x_i v2_l.
    def g_prod_and_gdg_prod_default(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2.unsqueeze(-2),
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_diagonal(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2,
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_additive(self, t, y, v1, v2):
        return self.g_prod(t, y, v1), 0.

    ########################################
    #              dg_ga_jvp               #
    ########################################

    # Computes: sum_{j,k,l} d g_{i,l} / d x_j g_{j,k} A_{k,l}.
    def dg_ga_jvp_column_sum_v1(self, t, y, a):
        requires_grad = torch.is_grad_enabled()
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
        requires_grad = torch.is_grad_enabled()
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

    def __init__(self, sde, drift='f', diffusion='g', prior_drift='h', diffusion_prod='g_prod',
                 drift_and_diffusion='f_and_g', drift_and_diffusion_prod='f_and_g_prod'):
        super(RenameMethodsSDE, self).__init__(noise_type=sde.noise_type, sde_type=sde.sde_type)
        self._base_sde = sde
        for name, value in zip(('f', 'g', 'h', 'g_prod', 'f_and_g', 'f_and_g_prod'),
                               (drift, diffusion, prior_drift, diffusion_prod, drift_and_diffusion,
                                drift_and_diffusion_prod)):
            try:
                setattr(self, name, getattr(sde, value))
            except AttributeError:
                pass


class SDEIto(BaseSDE):

    def __init__(self, noise_type):
        super(SDEIto, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.ito)


class SDEStratonovich(BaseSDE):

    def __init__(self, noise_type):
        super(SDEStratonovich, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.stratonovich)


# --- Backwards compatibility: v0.1.1. ---
class SDELogqp(BaseSDE):

    def __init__(self, sde):
        super(SDELogqp, self).__init__(noise_type=sde.noise_type, sde_type=sde.sde_type)
        self._base_sde = sde

        # Make this redirection a one-time cost.
        try:
            self._base_f = sde.f
            self._base_g = sde.g
            self._base_h = sde.h
        except AttributeError as e:
            # TODO: relax this requirement, and use f_and_g, f_and_g_prod, f_and_g_and_h and f_and_g_prod_and_h if
            #  they're available.
            raise AttributeError("If using logqp then drift, diffusion and prior drift must all be specified.") from e

        # Make this method selection a one-time cost.
        if sde.noise_type == NOISE_TYPES.diagonal:
            self.f = self.f_diagonal
            self.g = self.g_diagonal
            self.f_and_g = self.f_and_g_diagonal
        else:
            self.f = self.f_general
            self.g = self.g_general
            self.f_and_g = self.f_and_g_general

    def f_diagonal(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_diagonal(self, t, y: Tensor):
        y = y[:, :-1]
        g = self._base_g(t, y)
        g_logqp = y.new_zeros(size=(y.size(0), 1))
        return torch.cat([g, g_logqp], dim=1)

    def f_and_g_diagonal(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        g_logqp = y.new_zeros(size=(y.size(0), 1))
        return torch.cat([f, f_logqp], dim=1), torch.cat([g, g_logqp], dim=1)

    def f_general(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.batch_mvp(g.pinverse(), f - h)  # (batch_size, brownian_size).
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_general(self, t, y: Tensor):
        y = y[:, :-1]
        g = self._base_sde.g(t, y)
        g_logqp = y.new_zeros(size=(g.size(0), 1, g.size(-1)))
        return torch.cat([g, g_logqp], dim=1)

    def f_and_g_general(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.batch_mvp(g.pinverse(), f - h)  # (batch_size, brownian_size).
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        g_logqp = y.new_zeros(size=(g.size(0), 1, g.size(-1)))
        return torch.cat([f, f_logqp], dim=1), torch.cat([g, g_logqp], dim=1)
# ----------------------------------------

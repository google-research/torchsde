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

from ..settings import NOISE_TYPES, SDE_TYPES
from . import misc


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
        # Making these Python properties breaks `torch.jit.script`
        self.noise_type = noise_type
        self.sde_type = sde_type


class AdjointSDE(BaseSDE):
    """Base class for reverse-time adjoint SDE.

    Each forward SDE with different noise type has a different adjoint SDE.
    """

    def __init__(self, base_sde, noise_type):
        # `noise_type` must be supplied! Since the adjoint might have a different noise type than the original SDE.
        super(AdjointSDE, self).__init__(sde_type=base_sde.sde_type, noise_type=noise_type)
        self._base_sde = base_sde

    @abc.abstractmethod
    def f(self, t, y):
        pass

    @abc.abstractmethod
    def g(self, t, y):
        pass

    @abc.abstractmethod
    def h(self, t, y):
        pass

    @abc.abstractmethod
    def g_prod(self, t, y, v):
        pass

    @abc.abstractmethod
    def gdg_prod(self, t, y, v):
        pass


class ForwardSDE(BaseSDE):

    def __init__(self, base_sde):
        super(ForwardSDE, self).__init__(sde_type=base_sde.sde_type, noise_type=base_sde.noise_type)
        self._base_sde = base_sde

        # Register the core function. This avoids polluting the codebase with if-statements.
        self.g_prod = {
            NOISE_TYPES.diagonal: self.g_prod_diagonal,
            NOISE_TYPES.additive: self.g_prod_additive,
            NOISE_TYPES.scalar: self.g_prod_scalar,
            NOISE_TYPES.general: self.g_prod_general
        }[base_sde.noise_type]
        self.gdg_prod = {
            NOISE_TYPES.diagonal: self.gdg_prod_diagonal,
            NOISE_TYPES.additive: self._skip,
            NOISE_TYPES.scalar: self.gdg_prod_scalar,
            NOISE_TYPES.general: self.gdg_prod_general
        }
        self.gdg_jvp = {
            NOISE_TYPES.diagonal: self._skip,
            NOISE_TYPES.additive: self._skip,
            NOISE_TYPES.scalar: self._skip,
            NOISE_TYPES.general: self.gdg_jvp_v2
        }[base_sde.noise_type]

    def f(self, t, y):
        return self._base_sde.f(t, y)

    def g(self, t, y):
        return self._base_sde.g(t, y)

    def h(self, t, y):
        return self._base_sde.h(t, y)

    # g_prod functions.
    def g_prod_diagonal(self, t, y, v):
        return misc.seq_mul(self._base_sde.g(t, y), v)

    def g_prod_additive(self, t, y, v):
        return self.g_prod_general(t, y, v)

    def g_prod_scalar(self, t, y, v):
        return misc.seq_mul_bc(self._base_sde.g(t, y), v)

    def g_prod_general(self, t, y, v):
        return misc.seq_batch_mvp(ms=self._base_sde.g(t, y), vs=v)

    # gdg_prod functions.
    def gdg_prod_general(self, t, y, v):
        # TODO: Write this!
        raise NotImplemented

    def gdg_prod_diagonal(self, t, y, v):
        requires_grad = torch.is_grad_enabled()  # BP through solver.
        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) if not y_.requires_grad else y_ for y_ in y]
            val = self._base_sde.g(t, y)
            vjp_val = misc.grad(
                outputs=val,
                inputs=y,
                grad_outputs=misc.seq_mul(val, v),
                create_graph=requires_grad,
                allow_unused=True
            )
        return misc.convert_none_to_zeros(vjp_val, y)

    def gdg_prod_scalar(self, t, y, v):
        return self.gdg_prod_diagonal(t, y, v)

    # gdg_jvp functions.
    def gdg_jvp_compute(self, t, y, a):
        # Assumes `a` is anti-symmetric and `base_sde` is not of diagonal noise.
        requires_grad = torch.is_grad_enabled()  # BP through solver.
        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) if not y_.requires_grad else y_ for y_ in y]
            g_eval = self._base_sde.g(t, y)
            v = [torch.bmm(g_eval_, a_) for g_eval_, a_ in zip(g_eval, a)]
            gdg_jvp_eval = [
                misc.jvp(
                    outputs=[g_eval_[..., col_idx] for g_eval_ in g_eval],
                    inputs=y,
                    grad_inputs=[v_[..., col_idx] for v_ in v],
                    retain_graph=True,
                    create_graph=requires_grad,
                    allow_unused=True
                )
                for col_idx in range(g_eval[0].size(-1))
            ]
            gdg_jvp_eval = misc.seq_add(*gdg_jvp_eval)
        return misc.convert_none_to_zeros(gdg_jvp_eval, y)

    def gdg_jvp_v2(self, t, y, a):
        # Just like `gdg_jvp_compute`, but faster and more memory intensive.
        requires_grad = torch.is_grad_enabled()  # BP through solver.
        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) if not y_.requires_grad else y_ for y_ in y]
            g_eval = self._base_sde.g(t, y)
            v = [torch.bmm(g_eval_, a_) for g_eval_, a_ in zip(g_eval, a)]

            batch_size, d, m = g_eval[0].size()  # TODO: Relax this assumption.
            y_dup = [torch.repeat_interleave(y_, repeats=m, dim=0) for y_ in y]
            g_eval_dup = self._base_sde.g(t, y_dup)
            v_flat = [v_.transpose(1, 2).flatten(0, 1) for v_ in v]
            gdg_jvp_eval = misc.jvp(
                g_eval_dup, y_dup, grad_inputs=v_flat, create_graph=requires_grad, allow_unused=True
            )
            gdg_jvp_eval = misc.convert_none_to_zeros(gdg_jvp_eval, y)
            gdg_jvp_eval = [t.reshape(batch_size, m, d, m).permute(0, 2, 1, 3) for t in gdg_jvp_eval]
            gdg_jvp_eval = [t.diagonal(dim1=-2, dim2=-1).sum(-1) for t in gdg_jvp_eval]
        return gdg_jvp_eval

    def _skip(self, t, y, v):  # noqa
        return [0.] * len(y)


class TupleSDE(BaseSDE):

    def __init__(self, base_sde):
        super(TupleSDE, self).__init__(noise_type=base_sde.noise_type, sde_type=base_sde.sde_type)
        self._base_sde = base_sde

    def f(self, t, y):
        return self._base_sde.f(t, y[0]),

    def g(self, t, y):
        return self._base_sde.g(t, y[0]),

    def h(self, t, y):
        return self._base_sde.h(t, y[0]),


class RenameMethodsSDE(BaseSDE):

    def __init__(self, base_sde, drift='f', diffusion='g', prior_drift='h'):
        super(RenameMethodsSDE, self).__init__(noise_type=base_sde.noise_type, sde_type=base_sde.sde_type)
        self._base_sde = base_sde
        self.f = getattr(base_sde, drift)
        self.g = getattr(base_sde, diffusion)
        if hasattr(base_sde, prior_drift):
            self.h = getattr(base_sde, prior_drift)


class SDEIto(BaseSDE):

    def __init__(self, noise_type):
        super(SDEIto, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.ito)


class SDEStratonovich(BaseSDE):

    def __init__(self, noise_type):
        super(SDEStratonovich, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.stratonovich)

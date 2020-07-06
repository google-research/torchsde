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

import abc

import torch
from torch import nn

from torchsde.core import misc
from torchsde.core import settings


class BaseSDE(abc.ABC, nn.Module):
    """Base class for all SDEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(self, noise_type, sde_type):
        super(BaseSDE, self).__init__()
        if noise_type not in settings.NOISE_TYPES:
            raise ValueError(f"Expected noise type in {settings.NOISE_TYPES}, but found {noise_type}")
        if sde_type not in settings.SDE_TYPES:
            raise ValueError(f"Expected sde type in {settings.SDE_TYPES}, but found {sde_type}")
        self._noise_type = noise_type
        self._sde_type = sde_type

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, x):
        self._noise_type = x

    @property
    def sde_type(self):
        return self._sde_type


class SDEIto(BaseSDE):

    def __init__(self, noise_type):
        super(SDEIto, self).__init__(noise_type=noise_type, sde_type='ito')


class ForwardSDEIto(SDEIto):
    """Wrapper SDE for the forward pass.

    `g_prod` and `gdg_prod` are additional functions that high-order solvers will call.
    """

    def __init__(self, base_sde):
        super(ForwardSDEIto, self).__init__(noise_type=base_sde.noise_type)
        self._base_sde = base_sde

    def f(self, t, y):
        return self._base_sde.f(t, y)

    def g(self, t, y):
        return self._base_sde.g(t, y)

    def h(self, t, y):
        return self._base_sde.h(t, y)

    def g_prod(self, t, y, v):
        if self.noise_type == "diagonal":
            return misc.seq_mul(self._base_sde.g(t, y), v)
        elif self.noise_type == "scalar":
            return misc.seq_mul_bc(self._base_sde.g(t, y), v)
        return misc.seq_batch_mvp(ms=self._base_sde.g(t, y), vs=v)

    def gdg_prod(self, t, y, v):
        with torch.enable_grad():
            y = tuple(y_.detach().requires_grad_(True) if not y_.requires_grad else y_ for y_ in y)
            val = self._base_sde.g(t, y)
            val = misc.make_seq_requires_grad(val)
            vjp_val = torch.autograd.grad(
                outputs=val, inputs=y, grad_outputs=misc.seq_mul(val, v), create_graph=True, allow_unused=True)
            vjp_val = misc.convert_none_to_zeros(vjp_val, y)
        return vjp_val


class AdjointSDEIto(SDEIto):
    """Base class for reverse-time adjoint SDE.

    Each forward SDE with different noise type has a different adjoint SDE.
    """

    def __init__(self, base_sde, noise_type):
        super(AdjointSDEIto, self).__init__(noise_type=noise_type)
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


class SDEStratonovich(BaseSDE):

    def __init__(self, noise_type):
        super(SDEStratonovich, self).__init__(noise_type=noise_type, sde_type='stratonovich')


class TupleSDE(BaseSDE):

    def __init__(self, base_sde):
        super(TupleSDE, self).__init__(noise_type=base_sde.noise_type, sde_type=base_sde.sde_type)
        self._base_sde = base_sde

    def f(self, t, y):
        y, = y
        return self._base_sde.f(t, y),

    def g(self, t, y):
        y, = y
        return self._base_sde.g(t, y),

    def h(self, t, y):
        y, = y
        return self._base_sde.h(t, y),


class RenameMethodsSDE(BaseSDE):

    def __init__(self, base_sde, drift='f', diffusion='g', prior_drift='h'):
        super(RenameMethodsSDE, self).__init__(noise_type=base_sde.noise_type, sde_type=base_sde.sde_type)
        self._base_sde = base_sde
        self.f = getattr(base_sde, drift)
        self.g = getattr(base_sde, diffusion)
        self.h = getattr(base_sde, prior_drift)

    @property
    def noise_type(self):
        return self._base_sde.noise_type

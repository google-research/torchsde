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

"""Problems of different noise types.

Each example is of a particular noise type.

Ex1, Ex2, Ex3 from

Rackauckas, Christopher, and Qing Nie. "Adaptive methods for stochastic
differential equations via natural embeddings and rejection sampling with memory."
Discrete and continuous dynamical systems. Series B 22.7 (2017): 2731.

Ex4 is constructed to test schemes for SDEs with general noise and neural nets.
"""

import torch
from torch import nn

from torchsde import BaseSDE
from torchsde.settings import NOISE_TYPES, SDE_TYPES


class Ex1(BaseSDE):
    noise_type = NOISE_TYPES.diagonal

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex1, self).__init__(sde_type=sde_type, noise_type=Ex1.noise_type)
        self._nfe = 0

        # Use non-exploding initialization.
        sigma = torch.sigmoid(torch.randn(d))
        mu = -sigma ** 2 - torch.sigmoid(torch.randn(d))
        self.mu = nn.Parameter(mu, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

    def f(self, t, y):
        self._nfe += 1
        return self.mu * y

    def g(self, t, y):
        self._nfe += 1
        return self.sigma * y

    @property
    def nfe(self):
        return self._nfe


class Ex2(BaseSDE):
    noise_type = NOISE_TYPES.scalar

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex2, self).__init__(sde_type=sde_type, noise_type=Ex2.noise_type)
        self._nfe = 0
        self.p = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)

    def f(self, t, y):
        self._nfe += 1
        return -self.p ** 2. * torch.sin(y) * torch.cos(y) ** 3.

    def g(self, t, y):
        self._nfe += 1
        return (self.p * torch.cos(y) ** 2).unsqueeze(dim=1)

    @property
    def nfe(self):
        return self._nfe


class Ex3(BaseSDE):
    noise_type = NOISE_TYPES.additive

    def __init__(self, d, m, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex3, self).__init__(sde_type=sde_type, noise_type=Ex3.noise_type)
        self._nfe = 0
        self.m = m

        self.a = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)
        self.b = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)

    def f(self, t, y):
        self._nfe += 1
        return self.b / torch.sqrt(1. + t) - y / (2. + 2. * t)

    def g(self, t, y):
        self._nfe += 1
        fill_value = self.a * self.b / torch.sqrt(1. + t)
        return fill_value.unsqueeze(dim=0).unsqueeze(dim=-1).repeat(y.size(0), 1, self.m)

    @property
    def nfe(self):
        return self._nfe


class Ex4(BaseSDE):
    noise_type = NOISE_TYPES.general

    def __init__(self, d, m, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex4, self).__init__(sde_type=sde_type, noise_type=Ex4.noise_type)
        self._nfe = 0
        self.d = d
        self.m = m

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 3),
            nn.Softplus(),
            nn.Linear(3, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(d + 1, 3),
            nn.Softplus(),
            nn.Linear(3, d * m),
            nn.Sigmoid()
        )

    def f(self, t, y):
        self._nfe += 1
        ty = torch.cat((t.expand_as(y[:, :1]), y), dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        self._nfe += 1
        ty = torch.cat((t.expand_as(y[:, :1]), y), dim=1)
        return self.g_net(ty).reshape(-1, self.d, self.m)

    @property
    def nfe(self):
        return self._nfe

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

ExDiagonal, ExScalar, ExAdditive are examples 1-3 from
Rackauckas, Christopher, and Qing Nie. "Adaptive methods for stochastic
differential equations via natural embeddings and rejection sampling with memory."
Discrete and continuous dynamical systems. Series B 22.7 (2017): 2731.

Neural* all use simple neural networks.

BasicSDE1-4 are problems where the drift and diffusion may not depend on
trainable parameters.

CustomNamesSDE and CustomNamesSDELogqp are used to test the argument `names`.
"""

import torch
from torch import nn

from torchsde import BaseSDE, SDEIto
from torchsde.settings import NOISE_TYPES, SDE_TYPES


class ExDiagonal(BaseSDE):
    noise_type = NOISE_TYPES.diagonal

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(ExDiagonal, self).__init__(sde_type=sde_type, noise_type=ExDiagonal.noise_type)
        self._nfe = 0

        # Use non-exploding initialization.
        sigma = torch.sigmoid(torch.randn(d))
        mu = -sigma ** 2 - torch.sigmoid(torch.randn(d))
        self.mu = nn.Parameter(mu, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

        self.f = self.f_ito if sde_type == SDE_TYPES.ito else self.f_stratonovich

    def f_ito(self, t, y):
        self._nfe += 1
        return self.mu * y

    def f_stratonovich(self, t, y):
        self._nfe += 1
        return self.mu * y - .5 * (self.sigma ** 2) * y

    def g(self, t, y):
        self._nfe += 1
        return self.sigma * y

    def h(self, t, y):
        self._nfe += 1
        return torch.zeros_like(y)

    @property
    def nfe(self):
        return self._nfe


class ExScalar(BaseSDE):
    noise_type = NOISE_TYPES.scalar

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(ExScalar, self).__init__(sde_type=sde_type, noise_type=ExScalar.noise_type)
        self._nfe = 0
        self.p = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)

        self.f = self.f_ito if sde_type == SDE_TYPES.ito else self.f_stratonovich

    def f_ito(self, t, y):
        self._nfe += 1
        return -self.p ** 2. * torch.sin(y) * torch.cos(y) ** 3.

    def f_stratonovich(self, t, y):
        self._nfe += 1
        return torch.zeros_like(y)

    def g(self, t, y):
        self._nfe += 1
        return (self.p * torch.cos(y) ** 2).unsqueeze(dim=-1)

    def h(self, t, y):
        self._nfe += 1
        return torch.zeros_like(y)

    @property
    def nfe(self):
        return self._nfe


class ExAdditive(BaseSDE):
    noise_type = NOISE_TYPES.additive

    def __init__(self, d, m, sde_type=SDE_TYPES.ito, **kwargs):
        super(ExAdditive, self).__init__(sde_type=sde_type, noise_type=ExAdditive.noise_type)
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

    def h(self, t, y):
        self._nfe += 1
        return torch.zeros_like(y)

    @property
    def nfe(self):
        return self._nfe


class NeuralDiagonal(BaseSDE):
    noise_type = NOISE_TYPES.diagonal

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(NeuralDiagonal, self).__init__(sde_type=sde_type, noise_type=NeuralDiagonal.noise_type)

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 8),
            nn.Softplus(),
            nn.Linear(8, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(d + 1, 8),
            nn.Softplus(),
            nn.Linear(8, d),
            nn.Sigmoid()
        )

    def f(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return 0.1 * self.g_net(ty)  # small noise makes passing adjoint tests easier/possible

    def h(self, t, y):
        return torch.zeros_like(y)


class NeuralScalar(BaseSDE):
    noise_type = NOISE_TYPES.scalar

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(NeuralScalar, self).__init__(sde_type=sde_type, noise_type=NeuralScalar.noise_type)

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 8),
            nn.Softplus(),
            nn.Linear(8, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(d + 1, 8),
            nn.Softplus(),
            nn.Linear(8, d),
            nn.Sigmoid()
        )

    def f(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return 0.1 * self.g_net(ty).unsqueeze(-1)  # small noise makes passing adjoint tests easier/possible

    def h(self, t, y):
        return torch.zeros_like(y)


class NeuralAdditive(BaseSDE):
    noise_type = NOISE_TYPES.additive

    def __init__(self, d, m, sde_type=SDE_TYPES.ito, **kwargs):
        super(NeuralAdditive, self).__init__(sde_type=sde_type, noise_type=NeuralAdditive.noise_type)
        self.d = d
        self.m = m

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 8),
            nn.Softplus(),
            nn.Linear(8, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Softplus(),
            nn.Linear(8, d * m),
            nn.Sigmoid()
        )

    def f(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        return self.g_net(t.expand(y.size(0), 1)).view(y.size(0), self.d, self.m)

    def h(self, t, y):
        return torch.zeros_like(y)


class NeuralGeneral(BaseSDE):
    noise_type = NOISE_TYPES.general

    def __init__(self, d, m, sde_type=SDE_TYPES.ito, **kwargs):
        super(NeuralGeneral, self).__init__(sde_type=sde_type, noise_type=NeuralGeneral.noise_type)
        self.d = d
        self.m = m

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 8),
            nn.Softplus(),
            nn.Linear(8, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(d + 1, 8),
            nn.Softplus(),
            nn.Linear(8, d * m),
            nn.Sigmoid()
        )

    def f(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.g_net(ty).reshape(y.size(0), self.d, self.m)

    def h(self, t, y):
        return torch.zeros_like(y)


class BasicSDE1(SDEIto):
    def __init__(self, d=10):
        super(BasicSDE1, self).__init__(noise_type="diagonal")
        self.shared_param = nn.Parameter(torch.randn(1, d), requires_grad=True)
        self.no_grad_param = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param1 = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param2 = nn.Parameter(torch.randn(1, d), requires_grad=True)

    def f(self, t, y):
        return self.shared_param * torch.sin(y) * 0.2 + torch.cos(y ** 2.) * 0.1 + torch.cos(t) + self.no_grad_param * y

    def g(self, t, y):
        return torch.sigmoid(self.shared_param * torch.cos(y) * .3 + torch.sin(t)) + torch.sigmoid(
            self.no_grad_param * y) + 0.1

    def h(self, t, y):
        return torch.sigmoid(y)


class BasicSDE2(SDEIto):
    def __init__(self, d=10):
        super(BasicSDE2, self).__init__(noise_type="diagonal")
        self.shared_param = nn.Parameter(torch.randn(1, d), requires_grad=True)
        self.no_grad_param = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param1 = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param2 = nn.Parameter(torch.randn(1, d), requires_grad=True)

    def f(self, t, y):
        return self.shared_param * 0.2 + self.no_grad_param + torch.zeros_like(y)

    def g(self, t, y):
        return torch.sigmoid(self.shared_param * .3) + torch.sigmoid(self.no_grad_param) + torch.zeros_like(y) + 0.1

    def h(self, t, y):
        return torch.sigmoid(y)


class BasicSDE3(SDEIto):
    def __init__(self, d=10):
        super(BasicSDE3, self).__init__(noise_type="diagonal")
        self.shared_param = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.no_grad_param = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param1 = nn.Parameter(torch.randn(1, d), requires_grad=True)
        self.unused_param2 = nn.Parameter(torch.randn(1, d), requires_grad=False)

    def f(self, t, y):
        return self.shared_param * 0.2 + self.no_grad_param + torch.zeros_like(y)

    def g(self, t, y):
        return torch.sigmoid(self.shared_param * .3) + torch.sigmoid(self.no_grad_param) + torch.zeros_like(y) + 0.1

    def h(self, t, y):
        return torch.sigmoid(y)


class BasicSDE4(SDEIto):
    def __init__(self, d=10):
        super(BasicSDE4, self).__init__(noise_type="diagonal")
        self.shared_param = nn.Parameter(torch.randn(1, d), requires_grad=True)
        self.no_grad_param = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param1 = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param2 = nn.Parameter(torch.randn(1, d), requires_grad=True)

    def f(self, t, y):
        return torch.zeros_like(y).fill_(0.1)

    def g(self, t, y):
        return torch.sigmoid(torch.zeros_like(y)) + 0.1

    def h(self, t, y):
        return torch.sigmoid(y)


class CustomNamesSDE(SDEIto):
    def __init__(self):
        super(CustomNamesSDE, self).__init__(noise_type="diagonal")

    def forward(self, t, y):
        return y * t

    def g(self, t, y):
        return torch.sigmoid(t * y)


class CustomNamesSDELogqp(SDEIto):
    def __init__(self):
        super(CustomNamesSDELogqp, self).__init__(noise_type="diagonal")

    def forward(self, t, y):
        return y * t

    def g(self, t, y):
        return torch.sigmoid(t * y)

    def w(self, t, y):
        return y * t


class FGSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FGSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f(self, t, y):
        return -y

    def g(self, t, y):
        return y.unsqueeze(-1).sigmoid() * self.vector


class FAndGSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f_and_g(self, t, y):
        return -y, y.unsqueeze(-1).sigmoid() * self.vector


class GProdSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(GProdSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f(self, t, y):
        return -y

    def g_prod(self, t, y, v):
        return (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)


class FAndGProdSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGProdSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f_and_g_prod(self, t, y, v):
        return -y, (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)


class FAndGGProdSDE1(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGGProdSDE1, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f_and_g(self, t, y):
        return -y, y.unsqueeze(-1).sigmoid() * self.vector

    def g_prod(self, t, y, v):
        return (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)


class FAndGGProdSDE2(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGGProdSDE2, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f(self, t, y):
        return -y

    def f_and_g(self, t, y):
        return -y, y.unsqueeze(-1).sigmoid() * self.vector

    def g_prod(self, t, y, v):
        return (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)

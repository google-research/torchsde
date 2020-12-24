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

Ex1-4 all use simple neural networks.

BasicSDE1-4 are problems where the drift and diffusion may not depend on
trainable parameters.

CustomNamesSDE and CustomNamesSDELogqp are used to test the argument `names`.
"""

import torch
from torch import nn

from torchsde import BaseSDE, SDEIto
from torchsde.settings import NOISE_TYPES, SDE_TYPES


class Ex1(BaseSDE):
    noise_type = NOISE_TYPES.diagonal

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex1, self).__init__(sde_type=sde_type, noise_type=Ex1.noise_type)

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 3),
            nn.Softplus(),
            nn.Linear(3, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(d + 1, 3),
            nn.Softplus(),
            nn.Linear(3, d),
            nn.Sigmoid()
        )

    def f(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.g_net(ty)

    def h(self, t, y):
        return torch.zeros_like(y)


class Ex2(BaseSDE):
    noise_type = NOISE_TYPES.scalar

    def __init__(self, d, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex2, self).__init__(sde_type=sde_type, noise_type=Ex2.noise_type)

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 3),
            nn.Softplus(),
            nn.Linear(3, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(d + 1, 3),
            nn.Softplus(),
            nn.Linear(3, d),
            nn.Sigmoid()
        )

    def f(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.g_net(ty).view(y.size(0), self.d, 1)

    def h(self, t, y):
        return torch.zeros_like(y)


class Ex3(BaseSDE):
    noise_type = NOISE_TYPES.additive

    def __init__(self, d, m, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex3, self).__init__(sde_type=sde_type, noise_type=Ex3.noise_type)
        self.d = d
        self.m = m

        self.f_net = nn.Sequential(
            nn.Linear(d + 1, 3),
            nn.Softplus(),
            nn.Linear(3, d)
        )
        self.g_net = nn.Sequential(
            nn.Linear(1, 3),
            nn.Softplus(),
            nn.Linear(3, d * m),
            nn.Sigmoid()
        )

    def f(self, t, y):
        ty = torch.cat([t.expand(y.size(0), 1), y], dim=1)
        return self.f_net(ty)

    def g(self, t, y):
        return self.g_net(t.expand(y.size(0), 1)).view(y.size(0), self.d, self.m)

    def h(self, t, y):
        return torch.zeros_like(y)


class Ex4(BaseSDE):
    noise_type = NOISE_TYPES.general

    def __init__(self, d, m, sde_type=SDE_TYPES.ito, **kwargs):
        super(Ex4, self).__init__(sde_type=sde_type, noise_type=Ex4.noise_type)
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

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

"""Problems of various noise types."""
import torch
from torch import nn

from torchsde import SDEIto


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


class GeneralSDE(SDEIto):
    def __init__(self, d=10, m=3):
        super(GeneralSDE, self).__init__(noise_type="general")
        self.shared_param = nn.Parameter(torch.randn(1, d), requires_grad=True)
        self.no_grad_param = nn.Parameter(torch.randn(1, d, m), requires_grad=False)
        self.unused_param1 = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param2 = nn.Parameter(torch.randn(1, d), requires_grad=True)

    def f(self, t, y):
        return self.shared_param * torch.sin(y) * 0.2 + torch.cos(y ** 2.) * 0.1 + torch.cos(t)

    def g(self, t, y):
        return torch.sigmoid(y).unsqueeze(dim=2) * self.no_grad_param  # (batch_size, d, m).

    def h(self, t, y):
        return torch.sigmoid(y)


class AdditiveSDE(SDEIto):
    def __init__(self, d=10, m=3):
        super(AdditiveSDE, self).__init__(noise_type="additive")
        self.f_param = nn.Parameter(torch.randn(1, d), requires_grad=True)
        self.g_param = nn.Parameter(torch.sigmoid(torch.randn(1, d, m)), requires_grad=True)

    def f(self, t, y):
        return torch.sigmoid(y * self.f_param) * torch.sin(t)

    def g(self, t, y):
        return self.g_param.repeat(y.size(0), 1, 1)

    def h(self, t, y):
        return torch.sigmoid(y)


class ScalarSDE(AdditiveSDE):
    def __init__(self, d=10, m=3):
        super(ScalarSDE, self).__init__(d=d, m=m)
        self.g_param = nn.Parameter(torch.sigmoid(torch.randn(1, d, 1)), requires_grad=True)
        self.noise_type = "scalar"


class TupleSDE(SDEIto):
    def __init__(self, d=10):
        super(TupleSDE, self).__init__(noise_type="diagonal")
        self.shared_param = nn.Parameter(torch.randn(1, d), requires_grad=True)
        self.no_grad_param = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param1 = nn.Parameter(torch.randn(1, d), requires_grad=False)
        self.unused_param2 = nn.Parameter(torch.randn(1, d), requires_grad=True)

    def f(self, t, y):
        y, = y
        return (
            self.shared_param * torch.sin(y) * 0.2 +
            torch.sin(y ** 2.) * 0.1 +
            torch.cos(t) +
            self.no_grad_param * y,
        )

    def g(self, t, y):
        y, = y
        return torch.sigmoid(
            self.shared_param * torch.cos(y) * .3 + torch.sin(t)) + torch.sigmoid(self.no_grad_param * y),

    def h(self, t, y):
        y, = y
        return torch.sigmoid(y),


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

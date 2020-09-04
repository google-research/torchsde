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

"""Problems of different noise types with analytical solutions.

Ex1, Ex2, Ex3 from

Rackauckas, Christopher, and Qing Nie. "Adaptive methods for stochastic
differential equations via natural embeddings and rejection sampling with memory."
Discrete and continuous dynamical systems. Series B 22.7 (2017): 2731.

Ex4 is constructed to test schemes for SDEs with general noise.
"""

import torch
from torch import nn

from torchsde import BaseSDE
from torchsde.settings import NOISE_TYPES, SDE_TYPES


class Ex1(BaseSDE):
    def __init__(self, d=10, sde_type=SDE_TYPES.ito):
        super(Ex1, self).__init__(sde_type=sde_type, noise_type=NOISE_TYPES.diagonal)
        self.f = self.f_ito if sde_type == SDE_TYPES.ito else self.f_stratonovich
        self._nfe = 0

        # Use non-exploding initialization.
        sigma = torch.sigmoid(torch.randn(d))
        mu = -sigma ** 2 - torch.sigmoid(torch.randn(d))
        self.mu = nn.Parameter(mu, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

    def f_ito(self, t, y):
        self._nfe += 1
        return self.mu * y

    def f_stratonovich(self, t, y):
        self._nfe += 1
        return self.mu * y - .5 * (self.sigma ** 2) * y

    def g(self, t, y):
        self._nfe += 1
        return self.sigma * y

    def analytical_grad(self, y0, t, grad_output, bm):
        with torch.no_grad():
            ans = y0 * torch.exp((self.mu - self.sigma ** 2. / 2.) * t + self.sigma * bm(t))
            dmu = (grad_output * ans * t).mean(0)
            dsigma = (grad_output * ans * (-self.sigma * t + bm(t))).mean(0)
        return torch.cat((dmu, dsigma), dim=0)

    def analytical_sample(self, y0, ts, bm):
        assert ts[0] == 0
        with torch.no_grad():
            ans = [y0 * torch.exp((self.mu - self.sigma ** 2. / 2.) * t + self.sigma * bm(t)) for t in ts]
        return torch.stack(ans, dim=0)

    @property
    def nfe(self):
        return self._nfe


class Ex2(BaseSDE):
    def __init__(self, d=10, sde_type=SDE_TYPES.ito):
        super(Ex2, self).__init__(sde_type=sde_type, noise_type=NOISE_TYPES.diagonal)
        self.f = self.f_ito if sde_type == SDE_TYPES.ito else self.f_stratonovich
        self._nfe = 0
        self.p = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)

    def f_ito(self, t, y):
        self._nfe += 1
        return -self.p ** 2. * torch.sin(y) * torch.cos(y) ** 3.

    def f_stratonovich(self, t, y):
        self._nfe += 1
        return torch.zeros_like(y)

    def g(self, t, y):
        self._nfe += 1
        return self.p * torch.cos(y) ** 2

    def analytical_grad(self, y0, t, grad_output, bm):
        with torch.no_grad():
            wt = bm(t)
            dp = (grad_output * wt / (1. + (self.p * wt + torch.tan(y0)) ** 2.)).mean(0)
        return dp

    def analytical_sample(self, y0, ts, bm):
        assert ts[0] == 0
        with torch.no_grad():
            ans = [torch.atan(self.p * bm(t) + torch.tan(y0)) for t in ts]
        return torch.stack(ans, dim=0)

    @property
    def nfe(self):
        return self._nfe


class Ex1Scalar(Ex1):
    def __init__(self, d=10, sde_type=SDE_TYPES.ito):
        super(Ex1Scalar, self).__init__(d=d, sde_type=sde_type)
        self.noise_type = NOISE_TYPES.scalar

    def g(self, t, y):
        return super(Ex1Scalar, self).g(t, y).unsqueeze(2)


class Ex2Scalar(Ex2):
    def __init__(self, d=10, sde_type=SDE_TYPES.ito):
        super(Ex2Scalar, self).__init__(d=d, sde_type=sde_type)
        self.noise_type = NOISE_TYPES.scalar

    def g(self, t, y):
        return super(Ex2Scalar, self).g(t, y).unsqueeze(2)


# TODO: Make this a test problem for additive noise settings with decoupled m and d.
class Ex3(BaseSDE):
    def __init__(self, d=10, sde_type=SDE_TYPES.ito):
        super(Ex3, self).__init__(sde_type=sde_type, noise_type=NOISE_TYPES.diagonal)
        self._nfe = 0
        self.a = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)
        self.b = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)

    def f(self, t, y):
        self._nfe += 1
        return self.b / torch.sqrt(1. + t) - y / (2. + 2. * t)

    def g(self, t, y):
        self._nfe += 1
        return self.a * self.b / torch.sqrt(1. + t) + torch.zeros_like(y)  # Add dummy zero to make dimensions match.

    def analytical_grad(self, y0, t, grad_output, bm):
        with torch.no_grad():
            wt = bm(t)
            da = grad_output * self.b * wt / torch.sqrt(1. + t)
            db = grad_output * (t + self.a * wt) / torch.sqrt(1. + t)
            da = da.mean(0)
            db = db.mean(0)
        return torch.cat((da, db), dim=0)

    def analytical_sample(self, y0, ts, bm):
        assert ts[0] == 0
        with torch.no_grad():
            ans = [y0 / torch.sqrt(1 + t) + self.b * (t + self.a * bm(t)) / torch.sqrt(1 + t) for t in ts]
        return torch.stack(ans, dim=0)

    @property
    def nfe(self):
        return self._nfe


class Ex3Additive(Ex3):
    def __init__(self, d=10, sde_type=SDE_TYPES.ito):
        super(Ex3Additive, self).__init__(d=d, sde_type=sde_type)
        self.noise_type = NOISE_TYPES.additive

    def g(self, t, y):
        return torch.diag_embed(super(Ex3Additive, self).g(t=t, y=y))


def _column_wise_func(y, t, i):
    # This function is designed so that there are mixed partials.
    return (torch.cos(y * i + t * 0.1) * 0.2 +
            torch.sum(y, dim=-1, keepdim=True).cos() * 0.1)


class Ex4(BaseSDE):
    def __init__(self, d, m, sde_type=SDE_TYPES.ito):
        super(Ex4, self).__init__(sde_type=sde_type, noise_type=NOISE_TYPES.general)
        self.d = d
        self.m = m

    def f(self, t, y):
        return torch.sin(y) + t

    def g(self, t, y):
        return torch.stack([_column_wise_func(y, t, i) for i in range(self.m)], dim=-1)

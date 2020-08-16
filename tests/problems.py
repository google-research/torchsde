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

"""Problems with analytical solutions."""
import torch
from torch import nn

from torchsde import SDEIto, BaseSDE


class Ex1(SDEIto):
    def __init__(self, d=10):
        super(Ex1, self).__init__(noise_type="diagonal")
        self._nfe = 0
        # Use non-exploding initialization.
        sigma = torch.sigmoid(torch.randn(d))
        mu = -sigma ** 2 - torch.sigmoid(torch.randn(d))
        self.mu = nn.Parameter(mu, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

    def f(self, t, y):
        del t
        self._nfe += 1
        return self.mu * y

    def g(self, t, y):
        del t
        self._nfe += 1
        return self.sigma * y

    def analytical_grad(self, y0, t, grad_output, bm):
        with torch.no_grad():
            ans = y0 * torch.exp((self.mu - self.sigma ** 2. / 2.) * t + self.sigma * bm(t))
            dmu = (grad_output * ans * t).mean(0)
            dsigma = (grad_output * ans * (-self.sigma * t + bm(t))).mean(0)
        return torch.cat((dmu, dsigma), dim=0)

    def analytical_sample(self, y0, ts, bm):
        with torch.no_grad():
            ans = [y0]
            for next_t in ts[1:]:
                ans_ = y0 * torch.exp((self.mu - self.sigma ** 2. / 2.) * next_t + self.sigma * bm(next_t))
                ans.append(ans_)
        return torch.stack(ans, dim=0)

    @property
    def nfe(self):
        return self._nfe


class Ex2(BaseSDE):
    def __init__(self, d=10, sde_type='ito'):
        super(Ex2, self).__init__(noise_type="diagonal", sde_type=sde_type)
        self._nfe = 0
        self.p = nn.Parameter(torch.sigmoid(torch.randn(d)), requires_grad=True)

    def f(self, t, y):
        del t
        self._nfe += 1
        return -self.p ** 2. * torch.sin(y) * torch.cos(y) ** 3.

    def f_corr(self, t, y):
        del t
        self._nfe += 1
        return torch.zeros_like(y)

    def g(self, t, y):
        del t
        self._nfe += 1
        return self.p * torch.cos(y) ** 2

    def analytical_grad(self, y0, t, grad_output, bm):
        with torch.no_grad():
            wt = bm(t)
            dp = (grad_output * wt / (1. + (self.p * wt + torch.tan(y0)) ** 2.)).mean(0)
        return dp

    def analytical_sample(self, y0, ts, bm):
        with torch.no_grad():
            ans = [y0]
            for next_t in ts[1:]:
                wt = bm(next_t)
                ans.append(torch.atan(self.p * wt + torch.tan(y0)))
            ans = torch.stack(ans, dim=0)
        return ans

    @property
    def nfe(self):
        return self._nfe


class Ex3(SDEIto):
    def __init__(self, d=10):
        super(Ex3, self).__init__(noise_type="diagonal")
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
        del y0
        with torch.no_grad():
            wt = bm(t)
            da = grad_output * self.b * wt / torch.sqrt(1. + t)
            db = grad_output * (t + self.a * wt) / torch.sqrt(1. + t)
            da = da.mean(0)
            db = db.mean(0)
        return torch.cat((da, db), dim=0)

    def analytical_sample(self, y0, ts, bm):
        with torch.no_grad():
            ans = [y0]
            for t in ts[1:]:
                yt = y0 / torch.sqrt(1. + t) + self.b * (t + self.a * bm(t)) / torch.sqrt(1. + t)
                ans.append(yt)
            ans = torch.stack(ans, dim=0)
        return ans

    @property
    def nfe(self):
        return self._nfe


class Ex3Additive(Ex3):
    def __init__(self, d=10):
        super(Ex3Additive, self).__init__(d=d)
        self.noise_type = 'additive'

    def g(self, t, y):
        # Conform to additive noise SDE signature.
        gval = super(Ex3Additive, self).g(t=t, y=y)
        return torch.diag_embed(gval)

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

import torch
import torchsde
from torchsde import sdeint
import timeit


class SDE(torchsde.SDEIto):

    def __init__(self, mu, sigma):
        super().__init__(noise_type="diagonal")

        self.mu = mu
        self.sigma = sigma

    @torch.jit.export
    def f(self, t, y):
        return self.mu * y

    @torch.jit.export
    def g(self, t, y):
        return self.sigma * y


batch_size, d, m = 4, 1, 1  # State dimension d, Brownian motion dimension m.
geometric_bm = SDE(mu=0.5, sigma=1)

# Works for torch==1.6.0.
geometric_bm = torch.jit.script(geometric_bm)

y0 = torch.zeros(batch_size, d).fill_(0.1)  # Initial state.
ts = torch.linspace(0, 1, 20)


def time_func():
    ys = sdeint(geometric_bm, y0, ts, adaptive=False, dt=ts[1], options={'trapezoidal_approx': True})


print(timeit.Timer(time_func).timeit(number=100))

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

"""Temporary test for Stratonovich stuff.

This should be eventually refactored and the file should be removed.
"""

import torch
from torch import nn

import time
from torchsde._core.base_sde import ForwardSDE  # noqa
from torchsde import settings

torch.set_default_dtype(torch.float64)
cpu, gpu = torch.device('cpu'), torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu


class SDE(nn.Module):

    def __init__(self):
        super(SDE, self).__init__()
        self.noise_type = settings.NOISE_TYPES.general
        self.sde_type = settings.SDE_TYPES.stratonovich

    def f(self, t, y):
        return [torch.sin(y_) + t for y_ in y]

    def g(self, t, y):
        return [
            torch.stack([torch.cos(y_ ** 2 * i + t * 0.1) for i in range(m)], dim=-1)
            for y_ in y
        ]


batch_size, d, m = 3, 5, 12


def _make_inputs():
    t = torch.rand(()).to(device)
    y = [torch.randn(batch_size, d).to(device)]
    a = torch.randn(batch_size, m, m).to(device)
    a = [a - a.transpose(1, 2)]  # Anti-symmetric.
    sde = ForwardSDE(base_sde=SDE())
    return sde, t, y, a


def _time_function(func, reps=10):
    now = time.perf_counter()
    [func() for _ in range(reps)]
    return time.perf_counter() - now


def test_gdg_jvp():
    sde, t, y, a = _make_inputs()
    outs = sde.gdg_jvp_compute(t, y, a)
    outs_v2 = sde.gdg_jvp_v2(t, y, a)
    for out, out_v2 in zip(outs, outs_v2):
        assert torch.allclose(out, out_v2)


def check_efficiency():
    sde, t, y, a = _make_inputs()

    func1 = lambda: sde.gdg_jvp_compute(t, y, a)  # Linear in m.
    time_elapse = _time_function(func1)
    print(f'Time elapse for loop: {time_elapse:.4f}')

    func2 = lambda: sde.gdg_jvp_v2(t, y, a)  # Almost constant in m.
    time_elapse = _time_function(func2)
    print(f'Time elapse for duplicate: {time_elapse:.4f}')


test_gdg_jvp()
check_efficiency()

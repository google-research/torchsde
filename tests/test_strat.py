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

import time

import torch
from torch import nn

from torchsde import sdeint_adjoint, BrownianInterval
from torchsde import settings
from torchsde._core.base_sde import ForwardSDE, TupleSDE  # noqa

torch.manual_seed(1293819)
torch.set_default_dtype(torch.float64)
cpu, gpu = torch.device('cpu'), torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu
dtype = torch.get_default_dtype()
batch_size, d, m = 1, 2, 3
ts = torch.tensor([0.0, 0.2, 0.4]).to(device)
t0, t1 = ts[0], ts[-1]
y0 = torch.zeros(batch_size, d).fill_(0.1).to(device)


def _column_wise_func(y, t, i):
    # This function is designed so that there are mixed partials.
    # Also need to make sure this function doesn't drive the solution to explosion.
    return (torch.cos(y ** 2 * i + t * 0.1) * 0.2 +
            torch.tan(y[..., 0:1] * y[..., -2:-1]) * 0.3 +
            torch.sum(y ** 2, dim=-1, keepdim=True).cos() * 0.1)


class SDE(nn.Module):

    def __init__(self):
        super(SDE, self).__init__()
        self.noise_type = settings.NOISE_TYPES.general
        self.sde_type = settings.SDE_TYPES.stratonovich

    def f(self, t, y):
        return torch.sin(y) + t

    def g(self, t, y):
        return torch.stack([_column_wise_func(y, t, i) for i in range(m)], dim=-1)


def _batch_jacobian(output, input_):
    # Create batch of Jacobians for output of size (batch_size, d_o) and input of size (batch_size, d_i).
    assert output.dim() == input_.dim() == 2
    assert output.size(0) == input_.size(0)
    jacs = []
    for i in range(output.size(0)):  # batch_size.
        jac = []
        for j in range(output.size(1)):  # d_o.
            grad, = torch.autograd.grad(output[i, j], input_, retain_graph=True, allow_unused=True)
            grad = torch.zeros_like(input_[i]) if grad is None else grad[i].detach()
            jac.append(grad)
        jac = torch.stack(jac, dim=0)
        jacs.append(jac)
    return torch.stack(jacs, dim=0)


def _gdg_jvp_brute_force(sde, t, y, a):
    # Only returns the value for the first input-output pair.
    with torch.enable_grad():
        y = [y_.detach().requires_grad_(True) if not y_.requires_grad else y_ for y_ in y]
        g_eval = sde.g(t, y)
        v = [torch.bmm(g_eval_, a_) for g_eval_, a_ in zip(g_eval, a)]

        y0, g_eval0, v0 = y[0], g_eval[0], v[0]
        num_brownian = g_eval0.size(-1)
        jacobians_by_column = [_batch_jacobian(g_eval0[..., l], y0) for l in range(num_brownian)]
        return [
            sum(torch.bmm(jacobians_by_column[l], v0[..., l].unsqueeze(-1)).squeeze() for l in range(num_brownian))
        ]


def _make_inputs():
    t = torch.rand(()).to(device)
    y = [torch.randn(batch_size, d).to(device)]
    a = torch.randn(batch_size, m, m).to(device)
    a = [a - a.transpose(1, 2)]  # Anti-symmetric.
    sde = ForwardSDE(TupleSDE(SDE()))
    return sde, t, y, a


def test_gdg_jvp():
    sde, t, y, a = _make_inputs()
    outs_brute_force = _gdg_jvp_brute_force(sde, t, y, a)  # Reference.
    outs = sde.gdg_jvp_column_sum(t, y, a)
    outs_v2 = sde.gdg_jvp_column_sum_v2(t, y, a)
    for out_brute_force, out, out_v2 in zip(outs_brute_force, outs, outs_v2):
        assert torch.allclose(out_brute_force, out)
        assert torch.allclose(out_brute_force, out_v2)


def _time_function(func, reps=10):
    now = time.perf_counter()
    [func() for _ in range(reps)]
    return time.perf_counter() - now


def check_efficiency():
    sde, t, y, a = _make_inputs()

    func1 = lambda: sde.gdg_jvp_column_sum_v1(t, y, a)  # Linear in m.
    time_elapse = _time_function(func1)
    print(f'Time elapse for loop: {time_elapse:.4f}')

    func2 = lambda: sde.gdg_jvp_column_sum_v2(t, y, a)  # Almost constant in m.
    time_elapse = _time_function(func2)
    print(f'Time elapse for duplicate: {time_elapse:.4f}')


def test_adjoint():
    sde = SDE().to(device)
    bm = BrownianInterval(t0=t0, t1=t1, shape=(batch_size, m), dtype=dtype, device=device)

    def func(y0):
        ys = sdeint_adjoint(sde, y0, ts, bm, method='midpoint')
        return ys[-1].sum()

    y0_ = y0.clone().requires_grad_(True)
    torch.autograd.gradcheck(func, y0_, rtol=1e-4, atol=1e-3, eps=1e-8)


test_gdg_jvp()
check_efficiency()
test_adjoint()

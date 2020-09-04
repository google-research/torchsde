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

from torchsde import sdeint_adjoint, BrownianInterval
from torchsde._core.base_sde import ForwardSDE  # noqa
from torchsde.settings import SDE_TYPES
from .problems import Ex4

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)
cpu, gpu = torch.device('cpu'), torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu
dtype = torch.get_default_dtype()
batch_size, d, m = 1, 2, 3
ts = torch.tensor([0.0, 0.2, 0.4], device=device)
t0, t1 = ts[0], ts[-1]
y0 = torch.full((batch_size, d), 0.1, device=device)


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


def _dg_ga_jvp_brute_force(sde, t, y, a):
    with torch.enable_grad():
        y = y.detach().requires_grad_(True) if not y.requires_grad else y
        g = sde.g(t, y)
        ga = torch.bmm(g, a)

        num_brownian = g.size(-1)
        jacobians_by_column = [_batch_jacobian(g[..., l], y) for l in range(num_brownian)]
        return sum(torch.bmm(jacobians_by_column[l], ga[..., l].unsqueeze(-1)).squeeze() for l in range(num_brownian))


def _make_inputs():
    t = torch.rand((), device=device)
    y = torch.randn(batch_size, d, device=device)
    a = torch.randn(batch_size, m, m, device=device)
    a = a - a.transpose(1, 2)  # Anti-symmetric.
    sde = ForwardSDE(Ex4(d=d, m=m)).to(device)
    return sde, t, y, a


def test_dg_ga_jvp():
    sde, t, y, a = _make_inputs()
    outs_brute_force = _dg_ga_jvp_brute_force(sde, t, y, a)  # Reference.
    outs = sde.dg_ga_jvp_column_sum_v1(t, y, a)
    outs_v2 = sde.dg_ga_jvp_column_sum_v2(t, y, a)
    assert torch.is_tensor(outs_brute_force) and torch.is_tensor(outs) and torch.is_tensor(outs_v2)
    assert torch.allclose(outs_brute_force, outs)
    assert torch.allclose(outs_brute_force, outs_v2)


def _time_function(func, reps=10):
    now = time.perf_counter()
    [func() for _ in range(reps)]
    return time.perf_counter() - now


def check_efficiency():
    sde, t, y, a = _make_inputs()

    func1 = lambda: sde.dg_ga_jvp_column_sum_v1(t, y, a)  # Linear in m.
    time_elapse = _time_function(func1)
    print(f'Time elapse for loop: {time_elapse:.4f}')

    func2 = lambda: sde.dg_ga_jvp_column_sum_v2(t, y, a)  # Almost constant in m.
    time_elapse = _time_function(func2)
    print(f'Time elapse for duplicate: {time_elapse:.4f}')


def test_adjoint():
    sde = Ex4(d=d, m=m, sde_type=SDE_TYPES.stratonovich).to(device)
    bm = BrownianInterval(t0=t0, t1=t1, shape=(batch_size, m), dtype=dtype, device=device)

    def func(y0):
        ys = sdeint_adjoint(sde, y0, ts, bm, method='midpoint')
        return ys[-1].sum()

    y0_ = y0.clone().requires_grad_(True)
    torch.autograd.gradcheck(func, y0_, rtol=1e-4, atol=1e-3, eps=1e-8)


test_dg_ga_jvp()
check_efficiency()
test_adjoint()

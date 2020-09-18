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

import copy
import time

import torch
from torch import nn

from torchsde import sdeint_adjoint, BrownianInterval
from torchsde._core.base_sde import ForwardSDE  # noqa
from torchsde.settings import SDE_TYPES
from torchsde.types import Callable, TensorOrTensors, ModuleOrModules, Optional
from . import problems

torch.set_printoptions(precision=10)
torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)
cpu, gpu = torch.device('cpu'), torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu
dtype = torch.get_default_dtype()
batch_size, d, m = 5, 2, 3
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
    sde = ForwardSDE(problems.Ex4(d=d, m=m)).to(device)
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


def test_adjoint_inputs():
    sde = problems.Ex4(d=d, m=m, sde_type=SDE_TYPES.stratonovich).to(device)
    bm = BrownianInterval(t0=t0, t1=t1, shape=(batch_size, m), dtype=dtype, device=device)

    def func(inputs, modules):
        y0, sde = inputs[0], modules[0]
        ys = sdeint_adjoint(sde, y0, ts, bm, method='midpoint')
        return (ys[-1] ** 2).sum(dim=1).mean(dim=0)

    swiss_knife_gradcheck(func, y0, sde, eps=1e-7, rtol=1e-3, atol=1e-3, grad_inputs=True, gradgrad_inputs=True)


def test_adjoint_params():
    sde = problems.Ex5(d=d, m=m, sde_type=SDE_TYPES.stratonovich).to(device)
    bm = BrownianInterval(t0=t0, t1=t1, shape=(batch_size, m), dtype=dtype, device=device)

    def func(inputs, modules):
        """Outputs gradient norm squared."""
        y0, sde = inputs[0], modules[0]
        ys = sdeint_adjoint(sde, y0, ts, bm, method="midpoint")
        return (ys[-1] ** 2).sum(dim=1).mean(dim=0)

    swiss_knife_gradcheck(func, y0, sde, eps=1e-7, rtol=1e-3, atol=1e-3, grad_params=True, gradgrad_params=True)


def swiss_knife_gradcheck(func: Callable,
                          inputs: TensorOrTensors,
                          modules: Optional[ModuleOrModules] = (),
                          eps: float = 1e-6,
                          atol: float = 1e-5,
                          rtol: float = 1e-3,
                          grad_inputs=False,
                          gradgrad_inputs=False,
                          grad_params=False,
                          gradgrad_params=False):
    """Check grad and grad of grad wrt inputs and parameters of Modules.

    When `func` is vector-valued, the checks compare autodiff vjp against
    finite-difference vjp, where v is a sampled standard normal vector.

    This function is aimed to be as self-contained as possible so that could
    be copied/pasted across different projects.

    Args:
        func (callable): A Python function that takes in a sequence of tensors
            (inputs) and a sequence of nn.Module (modules), and outputs a tensor
            or a sequence of tensors.
        inputs (sequence of Tensors): The input tensors.
        modules (sequence of nn.Module): The modules whose parameter gradient
            needs to be tested.
        eps (float, optional): Magnitude of two-sided finite difference
            perturbation.
        atol (float, optional): Absolute tolerance.
        rtol (float, optional): Relative tolerance.
        grad_inputs (bool, optional): Check gradients wrt inputs if True.
        gradgrad_inputs (bool, optional): Check gradients of gradients wrt
            inputs if True.
        grad_params (bool, optional): Check gradients wrt differentiable
            parameters of modules if True.
        gradgrad_params (bool, optional): Check gradients of gradients wrt
            differentiable parameters of modules if True.

    Returns:
        None.
    """

    def convert_none_to_zeros(sequence, like_sequence):
        return [torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)]

    def flatten(sequence):
        return torch.cat([p.reshape(-1) for p in sequence]) if len(sequence) > 0 else torch.tensor([])

    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    if isinstance(modules, nn.Module):
        modules = (modules,)

    # Don't modify original objects.
    modules = tuple(copy.deepcopy(m) for m in modules)
    inputs = tuple(i.clone().requires_grad_() for i in inputs)

    func = _make_scalar_valued_func(func, inputs, modules)
    func_only_inputs = lambda *args: func(args, modules)  # noqa

    # Grad wrt inputs.
    if grad_inputs:
        torch.autograd.gradcheck(func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol)

    # Grad of grad wrt inputs.
    if gradgrad_inputs:
        torch.autograd.gradgradcheck(func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol)

    # Grad wrt params.
    if grad_params:
        params = [p for m in modules for p in m.parameters() if p.requires_grad]
        loss = func(inputs, modules)
        framework_grad = flatten(convert_none_to_zeros(torch.autograd.grad(loss, params, create_graph=True), params))

        numerical_grad = []
        for param in params:
            flat_param = param.reshape(-1)
            for i in range(len(flat_param)):
                flat_param[i] += eps  # In-place.
                plus_eps = func(inputs, modules).detach()
                flat_param[i] -= eps

                flat_param[i] -= eps
                minus_eps = func(inputs, modules).detach()
                flat_param[i] += eps

                numerical_grad.append((plus_eps - minus_eps) / (2 * eps))
                del plus_eps, minus_eps
        numerical_grad = torch.stack(numerical_grad)
        torch.testing.assert_allclose(numerical_grad, framework_grad, rtol=rtol, atol=atol)

    # Grad of grad wrt params.
    if gradgrad_params:
        # Define this outside `func_high_order` so that random tensors are generated only once.

        def func_high_order(inputs, modules):
            params = [p for m in modules for p in m.parameters() if p.requires_grad]
            grads = torch.autograd.grad(func(inputs, modules), params, create_graph=True, allow_unused=True)
            grads = tuple(grad for grad in grads if grad is not None)
            return grads

        swiss_knife_gradcheck(func_high_order, inputs, modules, rtol=rtol, atol=atol, eps=eps, grad_params=True)


def _make_scalar_valued_func(func, inputs, modules):
    outputs = func(inputs, modules)
    output_size = outputs.numel() if torch.is_tensor(outputs) else sum(o.numel() for o in outputs)

    if output_size > 1:
        grad_outputs = tuple(torch.randn_like(o) for o in outputs)

        def func_scalar_valued(inputs, modules):
            outputs = func(inputs, modules)
            return sum((output * grad_output).sum() for output, grad_output, in zip(outputs, grad_outputs))

        return func_scalar_valued

    return func

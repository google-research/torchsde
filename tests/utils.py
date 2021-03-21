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

import copy

import torch
from torch import nn

from torchsde.types import Callable, ModuleOrModules, Optional, TensorOrTensors


# These tolerances don't need to be this large. For gradients to match up in the Ito case, we typically need large
# values; not so much as in the Stratonovich case.
def assert_allclose(actual, expected, rtol=1e-3, atol=1e-2):
    if actual is None:
        assert expected is None
    else:
        torch.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def gradcheck(func: Callable,
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

    This function is aimed to be as self-contained as possible so that it could
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
                flat_param[i].data.add_(eps)
                plus_eps = func(inputs, modules).detach()
                flat_param[i].data.sub_(eps)

                flat_param[i].data.sub_(eps)
                minus_eps = func(inputs, modules).detach()
                flat_param[i].data.add_(eps)

                numerical_grad.append((plus_eps - minus_eps) / (2 * eps))
                del plus_eps, minus_eps
        numerical_grad = torch.stack(numerical_grad)
        torch.testing.assert_allclose(numerical_grad, framework_grad, rtol=rtol, atol=atol)

    # Grad of grad wrt params.
    if gradgrad_params:
        def func_high_order(inputs, modules):
            params = [p for m in modules for p in m.parameters() if p.requires_grad]
            grads = torch.autograd.grad(func(inputs, modules), params, create_graph=True, allow_unused=True)
            return tuple(grad for grad in grads if grad is not None)

        gradcheck(func_high_order, inputs, modules, rtol=rtol, atol=atol, eps=eps, grad_params=True)


def _make_scalar_valued_func(func, inputs, modules):
    outputs = func(inputs, modules)
    output_size = outputs.numel() if torch.is_tensor(outputs) else sum(o.numel() for o in outputs)

    if output_size > 1:
        # Define this outside `func_scalar_valued` so that random tensors are generated only once.
        grad_outputs = tuple(torch.randn_like(o) for o in outputs)

        def func_scalar_valued(inputs, modules):
            outputs = func(inputs, modules)
            return sum((output * grad_output).sum() for output, grad_output, in zip(outputs, grad_outputs))

        return func_scalar_valued

    return func

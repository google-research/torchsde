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

import functools
import operator
import warnings

import torch


def handle_unused_kwargs(obj, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn(f'{obj.__class__.__name__}: Unexpected arguments {unused_kwargs}')


def flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def convert_none_to_zeros(sequence, like_sequence):
    return [torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)]


def make_seq_requires_grad(sequence):
    return [p if p.requires_grad else p.detach().requires_grad_(True) for p in sequence]


def is_strictly_increasing(ts):
    return all(x < y for x, y in zip(ts[:-1], ts[1:]))


def is_nan(t):
    return torch.any(torch.isnan(t))


def seq_add(*seqs):
    return [sum(seq) for seq in zip(*seqs)]


def seq_mul(*seqs):
    return [functools.reduce(operator.mul, seq) for seq in zip(*seqs)]


def seq_mul_bc(*seqs):  # Supports broadcasting.
    soln = []
    for seq in zip(*seqs):
        cumprod = seq[0]
        for tensor in seq[1:]:
            # Insert dummy dims at the end of the tensor with fewer dims.
            num_missing_dims = cumprod.dim() - tensor.dim()
            if num_missing_dims > 0:
                new_size = tensor.size() + (1,) * num_missing_dims
                tensor = tensor.reshape(*new_size)
            elif num_missing_dims < 0:
                new_size = cumprod.size() + (1,) * num_missing_dims
                cumprod = cumprod.reshape(*new_size)
            cumprod = cumprod * tensor
        soln += [cumprod]
    return soln


def seq_sub(xs, ys):
    return [x - y for x, y in zip(xs, ys)]


def seq_sub_div(xs, ys, zs):
    return [_stable_div(x - y, z) for x, y, z in zip(xs, ys, zs)]


def _stable_div(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-7):
    y = torch.where(
        y.abs() > epsilon,
        y,
        torch.ones_like(y).fill_(epsilon) * y.sign()
    )
    return x / y


def seq_batch_mvp(ms, vs):
    return [batch_mvp(m, v) for m, v in zip(ms, vs)]


def batch_mvp(m, v):
    return torch.bmm(m, v.unsqueeze(-1)).squeeze(dim=-1)


def grad(outputs, inputs, **kwargs):
    outputs = make_seq_requires_grad(outputs)
    if torch.is_tensor(inputs):  # Workaround for PyTorch bug #39784.
        inputs = (inputs,)
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]

    _grad = torch.autograd.grad(outputs, inputs, **kwargs)
    return convert_none_to_zeros(_grad, inputs)


def jvp(outputs, inputs, grad_inputs=None, **kwargs):
    # `torch.autograd.functional.jvp` takes in `func` and requires re-evaluation.
    # The present implementation avoids this.
    outputs = make_seq_requires_grad(outputs)
    if torch.is_tensor(inputs):  # Workaround for PyTorch bug #39784.
        inputs = (inputs,)
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]

    dummy_outputs = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    vjp = torch.autograd.grad(outputs, inputs, grad_outputs=dummy_outputs, **kwargs)
    _jvp = torch.autograd.grad(vjp, dummy_outputs, grad_outputs=grad_inputs, **kwargs)
    return convert_none_to_zeros(_jvp, dummy_outputs)

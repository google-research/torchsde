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
import types
import warnings

import torch


def handle_unused_kwargs(obj, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn(f'{obj.__class__.__name__}: Unexpected arguments {unused_kwargs}')


def flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        p.reshape(-1) if p is not None else torch.zeros_like(q).reshape(-1)
        for p, q in zip(sequence, like_sequence)
    ]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def convert_none_to_zeros(sequence, like_sequence):
    return [torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)]


def make_seq_requires_grad(sequence):
    """Replace tensors in sequence that doesn't require gradients with tensors that requires gradients.

    Args:
        sequence: an Iterable of tensors.

    Returns:
        A list of tensors that all require gradients.
    """
    return [p if p.requires_grad else p.detach().requires_grad_(True) for p in sequence]


def is_increasing(t):
    return torch.all(torch.gt(t[1:], t[:-1]))


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


def seq_div(xs, ys):
    return [_stable_div(x, y) for x, y in zip(xs, ys)]


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


def is_seq_not_nested(x):
    if not _is_tuple_or_list(x):
        return False
    for xi in x:
        if _is_tuple_or_list(xi):
            return False
    return True


def _is_tuple_or_list(x):
    return isinstance(x, tuple) or isinstance(x, list)


def join(*iterables):
    """Return a generator which is an aggregate of all input generators.

    Useful for combining parameters of different `nn.Module` objects.
    """
    for iterable in iterables:
        assert isinstance(iterable, types.GeneratorType)
        yield from iterable


def batch_mvp(m, v):
    """Batched matrix vector product.

    Args:
        m: A tensor of size (batch_size, d, m).
        v: A tensor of size (batch_size, m).

    Returns:
        A tensor of size (batch_size, d).
    """
    v = v.unsqueeze(dim=-1)  # (batch_size, m, 1)
    mvp = torch.bmm(m, v)  # (batch_size, d, 1)
    mvp = mvp.squeeze(dim=-1)  # (batch_size, d)
    return mvp


def grad(inputs, **kwargs):
    # Workaround for PyTorch bug #39784
    if torch.is_tensor(inputs):
        inputs = (inputs,)
    _inputs = [torch.as_strided(input_, (), ()) for input_ in inputs]
    return torch.autograd.grad(inputs=inputs, **kwargs)

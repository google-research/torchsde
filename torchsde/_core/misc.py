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

import warnings

import torch


def assert_no_grad(names, maybe_tensors):
    for name, maybe_tensor in zip(names, maybe_tensors):
        if torch.is_tensor(maybe_tensor) and maybe_tensor.requires_grad:
            raise ValueError(f"Argument {name} must not require gradient.")


def handle_unused_kwargs(unused_kwargs, msg=None):
    if len(unused_kwargs) > 0:
        if msg is not None:
            warnings.warn(f"{msg}: Unexpected arguments {unused_kwargs}")
        else:
            warnings.warn(f"Unexpected arguments {unused_kwargs}")


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


def seq_sub(xs, ys):
    return [x - y for x, y in zip(xs, ys)]


def batch_mvp(m, v):
    return torch.bmm(m, v.unsqueeze(-1)).squeeze(dim=-1)


def vjp(outputs, inputs, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    _vjp = torch.autograd.grad(outputs, inputs, **kwargs)
    return convert_none_to_zeros(_vjp, inputs)


def jvp(outputs, inputs, grad_inputs=None, **kwargs):
    # Unlike `torch.autograd.functional.jvp`, this function avoids repeating forward computation.
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    dummy_outputs = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    _vjp = torch.autograd.grad(outputs, inputs, grad_outputs=dummy_outputs, create_graph=True, allow_unused=True)
    _vjp = make_seq_requires_grad(convert_none_to_zeros(_vjp, inputs))

    _jvp = torch.autograd.grad(_vjp, dummy_outputs, grad_outputs=grad_inputs, **kwargs)
    return convert_none_to_zeros(_jvp, dummy_outputs)


def flat_to_shape(tensor, shapes, length=()):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tensor_list

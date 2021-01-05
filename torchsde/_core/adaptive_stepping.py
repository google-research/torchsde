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

from . import misc
from ..types import TensorOrTensors


def update_step_size(error_estimate, prev_step_size, safety=0.9, facmin=0.2, facmax=1.4, prev_error_ratio=None):
    """Adaptively propose the next step size based on estimated errors."""
    if error_estimate > 1:
        pfactor = 0
        ifactor = 1 / 1.5  # 1 / 5
    else:
        pfactor = 0.13
        ifactor = 1 / 4.5  # 1 / 15

    error_ratio = safety / error_estimate
    if prev_error_ratio is None:
        prev_error_ratio = error_ratio
    factor = error_ratio ** ifactor * (error_ratio / prev_error_ratio) ** pfactor
    if error_estimate <= 1:
        prev_error_ratio = error_ratio
        facmin = 1.0
    factor = min(facmax, max(facmin, factor))
    new_step_size = prev_step_size * factor
    return new_step_size, prev_error_ratio


def compute_error(y11: TensorOrTensors, y12: TensorOrTensors, rtol, atol, eps=1e-7):
    """Computer error estimate.

    Args:
        y11: A tensor or a sequence of tensors obtained with a full update.
        y12: A tensor or a sequence of tensors obtained with two half updates.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        eps: A small constant to avoid division by zero.

    Returns:
        A float for the aggregated error estimate.
    """
    if torch.is_tensor(y11):
        y11 = (y11,)
    if torch.is_tensor(y12):
        y12 = (y12,)
    tol = [
        (rtol * torch.max(torch.abs(y11_), torch.abs(y12_)) + atol).clamp_min(eps)
        for y11_, y12_ in zip(y11, y12)
    ]
    error_estimate = _rms(
        [(y11_ - y12_) / tol_ for y11_, y12_, tol_ in zip(y11, y12, tol)], eps
    )
    assert not misc.is_nan(error_estimate), (
        'Found nans in the error estimate. Try increasing the tolerance or regularizing the dynamics.'
    )
    return error_estimate.detach().cpu().item()


def _rms(x, eps=1e-7):
    if torch.is_tensor(x):
        return torch.sqrt((x ** 2.).sum() / x.numel()).clamp_min(eps)
    else:
        return torch.sqrt(sum((x_ ** 2.).sum() for x_ in x) / sum(x_.numel() for x_ in x)).clamp_min(eps)

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

import os

import numpy as np
from scipy import stats
import torch


def to_numpy(*args):
    if len(args) == 1:
        arg = args[0]
        if not isinstance(arg, torch.Tensor):
            raise ValueError('Input should be one or a list of torch tensors.')
        return _to_numpy_single(args[0])
    else:
        if not all(isinstance(arg, torch.Tensor) for arg in args):
            raise ValueError('Input should be one or a list of torch tensors.')
        return tuple(_to_numpy_single(arg) for arg in args)


def _to_numpy_single(arg):
    return arg.detach().cpu().numpy()


def compute_mse(x, y, normdim=1, meandim=0):
    normdiff = torch.norm(x - y, dim=normdim)
    sqrnormdiff = normdiff ** 2.
    mse = sqrnormdiff.mean(dim=meandim)
    assert mse.size() == ()
    return _to_numpy_single(mse)


def makedirs(*dirs):
    for d in dirs:
        assert isinstance(d, str)
        if not os.path.exists(d):
            os.makedirs(d)


def log(x):
    if not isinstance(x, np.ndarray):
        return np.log(np.array(x))
    return np.log(x)


def half_log(x):
    return .5 * log(x)


def regress_slope(x, y):
    k, _, _, _, _ = stats.linregress(x, y)
    return k

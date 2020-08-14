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

from typing import Union

import torch
from torchsde._brownian_lib import BrownianPath as _BrownianPath  # noqa

from .._brownian import utils  # noqa
from .._brownian import base_brownian # noqa


class BrownianPath(base_brownian.BaseBrownian):
    """Fast Brownian motion with all queries stored in a std::map and uses local search.

    Trades in memory for speed.

    To use:
    >>> bm = BrownianPath(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]])
    """

    def __init__(self,
                 t0: Union[float, torch.Tensor],
                 w0: torch.Tensor,
                 **kwargs):  # noqa
        super(BrownianPath, self).__init__()
        if not utils.is_scalar(t0):
            raise ValueError('Initial time t0 should be a float or 0-d torch.Tensor.')
        self._bm = _BrownianPath(t0=t0, w0=w0)

    def __call__(self, ta, tb):
        return self.call(tb) - self.call(ta)

    def call(self, t):
        return self._bm(t)

    def __repr__(self):
        return repr(self._bm)

    def insert(self, t, w):
        self._bm.insert(t, w)

    def to(self, *args, **kwargs):
        cache = self._bm.get_cache()
        for k, v in cache.items():
            cache[k] = v.to(*args, **kwargs)
        self._bm = _BrownianPath(data=cache)

    @property
    def dtype(self):
        return self._bm.get_w_head().dtype

    @property
    def device(self):
        return self._bm.get_w_head().device

    @property
    def shape(self):
        return self._bm.get_w_head().shape

    def get_cache(self):
        return self._bm.get_cache()

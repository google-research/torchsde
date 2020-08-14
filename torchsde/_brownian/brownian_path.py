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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import blist
import numpy as np
import torch

from torchsde._brownian import utils
from torchsde._brownian.base_brownian import Brownian

from typing import Union


class BrownianPath(Brownian):
    """Fast Brownian motion with all queries stored in a list and uses local search.

    Trades in memory for speed.

    To use:
    >>> bm = BrownianPath(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]])
    """

    def __init__(self, t0: Union[float, torch.Tensor], w0: torch.Tensor, window_size: int = 8):
        """Initialize Brownian path.

        Args:
            t0: Initial time.
            w0: Initial state.
            window_size: Size of the window around the last query for local search.
        """
        super(BrownianPath, self).__init__()
        if not utils.is_scalar(t0):
            raise ValueError('Initial time t0 should be a float or 0-d torch.Tensor.')

        t0 = float(t0)
        self._ts = blist.blist()
        self._ws = blist.blist()
        self._ts.append(t0)
        self._ws.append(w0)

        self._last_idx = 0
        self._window_size = window_size

    def __call__(self, ta, tb):
        return self.call(tb) - self.call(ta)

    def call(self, t):
        t = float(t)
        if t == self._ts[-1]:
            idx = len(self._ts) - 1
            w = self._ws[idx]
            found = True
        elif t == self._ts[0]:
            idx = 0
            w = self._ws[idx]
            found = True
        elif t > self._ts[-1]:
            idx = len(self._ts)
            t0 = self._ts[-1]
            t1 = t
            dw = torch.randn_like(self._ws[0]) * math.sqrt(t1 - t0)
            w = self._ws[-1] + dw
            found = False
        elif t < self._ts[0]:
            idx = 0
            t0 = t
            t1 = self._ts[0]
            dw = torch.randn_like(self._ws[0]) * math.sqrt(t1 - t0)
            w = self._ws[0] - dw
            found = False
        else:
            # Try local neighborhood of last query.
            left_idx = max(0, self._last_idx - self._window_size)
            right_idx = min(len(self._ts), self._last_idx + self._window_size)
            idx, w, found = utils.search(self._ts[left_idx:right_idx], self._ws[left_idx:right_idx], t)
            if w is None:  # t not within range of local neighborhood.
                # t must be within range of self._ts. This is ensured by the entering logic.
                idx, w, found = utils.search(self._ts, self._ws, t)
            else:  # Convert idx to be in full range.
                idx = idx + left_idx

        if not found:
            self._ts.insert(idx, t)
            self._ws.insert(idx, w)
        self._last_idx = idx
        return w

    def insert(self, t, w):
        """Insert scalar time and tensor Brownian motion state into list.

        The method silently replaces the original value if t is already in the list, and returns the old value.
        Otherwise returns None.

        The method should only be used for testing purposes.
        """
        t = float(t)
        if t > self._ts[-1]:
            idx = len(self._ts)
            old = None
        elif t < self._ts[0]:
            idx = 0
            old = None
        else:
            # TODO: Replace with `torch.searchsorted` when torch==1.7.0 releases.
            #  Also need to make sure we use tensor dt.
            idx = np.searchsorted(self._ts, t)
            if t == self._ts[idx]:
                old = self._ws[idx]
            else:
                old = None

        self._ts.insert(idx, t)
        self._ws.insert(idx, w)
        self._last_idx = idx
        return old

    def __repr__(self):
        return f"BrownianPath(t0={self._ts[0]:.3f}, t1={self._ts[-1]:.3f})"

    def to(self, *args, **kwargs):
        self._ws = utils.blist_to(self._ws, *args, **kwargs)

    @property
    def dtype(self):
        return self._ws[0].dtype

    @property
    def device(self):
        return self._ws[0].device

    @property
    def shape(self):
        return self._ws[0].size()

    def __len__(self):
        return len(self._ts)

    def get_cache(self):
        return {'ts': self._ts, 'ws': self._ws}

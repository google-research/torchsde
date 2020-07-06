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

import copy
import math

import blist
import numpy as np
import torch
from numpy.random import SeedSequence

from torchsde.brownian import base
from torchsde.brownian import utils


class BrownianTree(base.Brownian):
    """Brownian tree with fixed entropy.

    Trades in speed for memory.

    To use:
    >>> bm = BrownianTree(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]])
    """

    def __init__(self, t0, w0: torch.Tensor, t1=None, w1=None, entropy=None, tol=1e-6, pool_size=24, cache_depth=9,
                 safety=None):
        """Initialize the Brownian tree.

        The random value generation process exploits the parallel random number paradigm and uses
        `numpy.random.SeedSequence`. The default generator is PCG64 (used by `default_rng`).

        Args:
            t0: float or torch.Tensor for initial time.
            t1: float or torch.Tensor for terminal time.
            w0: torch.Tensor for initial state.
            w1: torch.Tensor for terminal state.
            entropy: Global seed, defaults to `None` for random entropy.
            tol: Error tolerance before the binary search is terminated; the search depth ~ log2(tol).
            pool_size: Size of the pooled entropy; should be larger than max depth of queries.
                This parameter affects the query speed significantly.
            cache_depth: Depth of the tree to cache values. This parameter affects the query speed significantly.
            safety: Small float representing some time increment before t0 and after t1. In practice, we don't let t0
                and t1 of the Brownian tree be the start and terminal times of the solutions. This is to avoid issues
                related to 1) finite precision, and 2) adaptive solver querying time points beyond initial and terminal
                times.
        """
        super(BrownianTree, self).__init__()
        if not utils.is_scalar(t0):
            raise ValueError('Initial time t0 should be a float or 0-d torch.Tensor.')

        if t1 is None:
            t1 = t0 + 1.0
        if not utils.is_scalar(t1):
            raise ValueError('Terminal time t1 should be a float or 0-d torch.Tensor.')
        if t0 > t1:
            raise ValueError(f'Initial time {t0} should be less than terminal time {t1}.')
        t0, t1 = float(t0), float(t1)

        if w1 is None:
            w1 = w0 + torch.randn_like(w0) * math.sqrt(t1 - t0)

        self._t0 = t0
        self._t1 = t1

        self._entropy = entropy
        self._tol = tol
        self._pool_size = pool_size
        self._cache_depth = cache_depth

        # Boundary guards.
        if safety is None:
            safety = 0.1 * (t1 - t0)
        t00 = t0 - safety
        t11 = t1 + safety

        self._ts_prev = blist.blist()
        self._ws_prev = blist.blist()
        self._ts_prev.extend([t00, t0])
        self._ws_prev.extend([w0 + torch.randn_like(w0) * math.sqrt(t0 - t00), w0])

        self._ts_post = blist.blist()
        self._ws_post = blist.blist()
        self._ts_post.extend([t1, t11])
        self._ws_post.extend([w1, w1 + torch.randn_like(w1) * math.sqrt(t11 - t1)])

        # Cache.
        ts, ws, seeds = _create_cache(t0=t0, t1=t1, w0=w0, w1=w1, entropy=entropy, pool_size=pool_size, k=cache_depth)
        self._ts = ts
        self._ws = ws
        self._seeds = seeds

        self._last_depth = None

    def __call__(self, t):
        t = float(t)
        if t <= self._t0:
            return utils.search_and_insert(ts=self._ts_prev, ws=self._ws_prev, t=t)
        if t >= self._t1:
            return utils.search_and_insert(ts=self._ts_post, ws=self._ws_post, t=t)

        i = np.searchsorted(self._ts, t)
        parent = copy.copy(self._seeds[i - 1])  # Spawn modifies the seed.
        t0, t1 = self._ts[i - 1], self._ts[i]
        w0, w1 = self._ws[i - 1], self._ws[i]
        wt, depth = _binary_search(t0=t0, t1=t1, w0=w0, w1=w1, t=t, parent=parent, tol=self._tol)
        self._last_depth = depth
        return wt

    @property
    def last_depth(self):
        return self._last_depth

    def __repr__(self):
        return (
            f"BrownianTree(t0={self._t0:.3f}, t1={self._t1:.3f}, "
            f"entropy={self._entropy}, tol={self._tol}, pool_size={self._pool_size}, "
            f"cache_depth={self._cache_depth})"
        )

    def to(self, *args, **kwargs):
        self._ws_prev = _list_to(self._ws_prev, *args, **kwargs)
        self._ws_post = _list_to(self._ws_post, *args, **kwargs)
        self._ws = _list_to(self._ws, *args, **kwargs)

    @property
    def dtype(self):
        return self._ws[0].dtype

    @property
    def device(self):
        return self._ws[0].device

    @property
    def size(self):
        return self._ws[0].size()

    def __len__(self):
        return len(self._ts) + len(self._ts_prev) + len(self._ts_post)


def _binary_search(t0, t1, w0, w1, t, parent, tol):
    seedv, seedl, seedr = parent.spawn(3)
    t_mid = (t0 + t1) / 2
    w_mid = utils.brownian_bridge(t0=t0, t1=t1, w0=w0, w1=w1, t=t_mid, seed=seedv)
    depth = 0

    while abs(t - t_mid) > tol:
        if t < t_mid:
            t0, t1 = t0, t_mid
            w0, w1 = w0, w_mid
            parent = seedl
        else:
            t0, t1 = t_mid, t1
            w0, w1 = w_mid, w1
            parent = seedr

        seedv, seedl, seedr = parent.spawn(3)
        t_mid = (t0 + t1) / 2
        w_mid = utils.brownian_bridge(t0=t0, t1=t1, w0=w0, w1=w1, t=t_mid, seed=seedv)
        depth += 1

    return w_mid, depth


def _create_cache(t0, t1, w0, w1, entropy, pool_size, k):
    ts = [t0, t1]
    ws = [w0, w1]

    parent = SeedSequence(entropy=entropy, pool_size=pool_size)
    seeds = [parent]

    for level in range(1, k + 1):
        new_ts = []
        new_ws = []
        new_seeds = []
        for i, parent in enumerate(seeds):
            seedv, seedl, seedr = parent.spawn(3)
            new_seeds.extend([seedl, seedr])

            t0, t1 = ts[i], ts[i + 1]
            w0, w1 = ws[i], ws[i + 1]
            t = (t0 + t1) / 2
            w = utils.brownian_bridge(t0=t0, t1=t1, w0=w0, w1=w1, t=t, seed=seedv)
            new_ts.extend([ts[i], t])
            new_ws.extend([ws[i], w])

        new_ts.append(ts[-1])
        new_ws.append(ws[-1])
        ts = new_ts
        ws = new_ws
        seeds = new_seeds

    return ts, ws, seeds


def _list_to(l, *args, **kwargs):
    return [li.to(*args, **kwargs) for li in l]

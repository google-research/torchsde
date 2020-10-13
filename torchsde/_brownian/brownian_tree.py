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

import bisect
import copy
import math
import warnings

import blist
import torch
from numpy.random import SeedSequence

from . import base_brownian
from . import utils
from .._core.misc import handle_unused_kwargs
from ..settings import LEVY_AREA_APPROXIMATIONS
from ..types import Scalar, Optional, Tensor


class BrownianTree(base_brownian.BaseBrownian):
    """Brownian tree with fixed entropy.

    Trades in speed for memory.

    To use:
    >>> bm = BrownianTree(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]])
    """

    def __init__(self,
                 t0: Scalar,
                 w0: Tensor,
                 t1: Optional[Scalar] = None,
                 w1: Optional[Tensor] = None,
                 entropy: Optional[int] = None,
                 tol: float = 1e-6,
                 pool_size: int = 24,
                 cache_depth: int = 9,
                 safety: Optional[float] = None,
                 levy_area_approximation: str = LEVY_AREA_APPROXIMATIONS.none,
                 **unused_kwargs):
        """Initialize the Brownian tree.

        The random value generation process exploits the parallel random number
        paradigm and uses `numpy.random.SeedSequence`. The default generator is
        PCG64 (used by `default_rng`).

        Args:
            t0 (float or Tensor): Initial time.
            w0 (sequence of Tensor): Initial state.
            t1 (float or Tensor): Terminal time.
            w1 (sequence of Tensor): Terminal state.
            entropy (int): Global seed, defaults to `None` for random entropy.
            tol (float or Tensor): Error tolerance before the binary search is
                terminated; the search depth ~ log2(tol).
            pool_size (int): Size of the pooled entropy; should be larger than
                max depth of queries. This parameter affects the query speed
                significantly.
            cache_depth (int): Depth of the tree to cache values. This parameter
                affects the query speed significantly.
            safety (float): Small float representing some time increment before
                t0 and after t1. In practice, we don't let t0 and t1 of the
                Brownian tree be the start and terminal times of the solutions.
                This is to avoid issues related to 1) finite precision, and 2)
                adaptive solver querying time points beyond initial and
                terminal times.
            levy_area_approximation (str): Whether to also approximate Levy
                area. Defaults to None. Valid options are either 'none',
                'space-time', 'davie' or 'foster', corresponding to
                approximation type. This is needed for some higher-order SDE
                solvers.
        """
        handle_unused_kwargs(unused_kwargs, msg=self.__class__.__name__)
        del unused_kwargs

        super(BrownianTree, self).__init__()
        if not utils.is_scalar(t0):
            raise ValueError('Initial time t0 should be a float or 0-d torch.Tensor.')
        if t1 is None:
            t1 = t0 + 1.0
        if not utils.is_scalar(t1):
            raise ValueError('Terminal time t1 should be a float or 0-d torch.Tensor.')
        if t0 > t1:
            raise ValueError(f'Initial time {t0} should be less than terminal time {t1}.')

        if levy_area_approximation != LEVY_AREA_APPROXIMATIONS.none:
            raise ValueError(
                "Only BrownianInterval currently supports levy_area_approximation for values other than 'none'."
            )

        t0, t1 = float(t0), float(t1)

        generator, parent = SeedSequence(entropy=entropy, pool_size=pool_size).spawn(2)
        w1_seed, w00_seed, w11_seed = generator.generate_state(3)

        if w1 is None:
            w1 = w0 + utils.randn_like(ref=w0, seed=w1_seed) * math.sqrt(t1 - t0)

        self._t0 = t0
        self._t1 = t1

        self._entropy = entropy
        self._tol = tol
        self._pool_size = pool_size
        self._cache_depth = cache_depth
        self.levy_area_approximation = levy_area_approximation

        # Boundary guards.
        if safety is None:
            safety = 0.1 * (t1 - t0)
        t00 = t0 - safety
        t11 = t1 + safety

        self._ts_prev = blist.blist()
        self._ws_prev = blist.blist()
        self._ts_prev.extend([t00, t0])
        w00 = w0 + utils.randn_like(ref=w0, seed=w00_seed) * math.sqrt(t0 - t00)
        self._ws_prev.extend([w00, w0])

        self._ts_post = blist.blist()
        self._ws_post = blist.blist()
        self._ts_post.extend([t1, t11])
        w11 = w1 + utils.randn_like(ref=w0, seed=w11_seed) * math.sqrt(t11 - t1)
        self._ws_post.extend([w1, w11])

        # Cache.
        ts, ws, seeds = _create_cache(t0=t0, t1=t1, w0=w0, w1=w1, parent=parent, k=cache_depth)
        self._ts = ts
        self._ws = ws
        self._seeds = seeds

        self._last_depth = None

    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        if tb is None:
            W = self.call(ta)
        else:
            W = self.call(tb) - self.call(ta)
        U = None
        A = None

        if return_U:
            if return_A:
                return W, U, A
            else:
                return W, U
        else:
            if return_A:
                return W, A
            else:
                return W

    def call(self, t):
        t = float(t)
        if t < self._t0:
            warnings.warn(f"Should have t>=t0 but got t={t} and t0={self._t0}.")
            t = self._t0
        if t > self._t1:
            warnings.warn(f"Should have t<=t1 but got t={t} and t1={self._t1}.")
            t = self._t1

        i = bisect.bisect_left(self._ts, t)
        if i < len(self._ts) and t == self._ts[i]:  # `t` in cache.
            return self._ws[i]

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
        raise AttributeError(f"BrownianTree does not support the method `to`.")

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
        return len(self._ts) + len(self._ts_prev) + len(self._ts_post)

    def get_cache(self):
        curr = {k: v for k, v in zip(self._ts, self._ws)}
        prev = {k: v for k, v in zip(self._ts_prev, self._ws_prev)}
        post = {k: v for k, v in zip(self._ts_post, self._ws_post)}
        return curr, prev, post


def _binary_search(t0, t1, w0, w1, t, parent, tol):
    seed_v, seed_l, seed_r = parent.spawn(3)
    seed_v, = seed_v.generate_state(1)

    t_mid = (t0 + t1) / 2
    w_mid = utils.brownian_bridge(t0=t0, t1=t1, w0=w0, w1=w1, t=t_mid, seed=seed_v)
    depth = 0

    while abs(t - t_mid) > tol:
        if t < t_mid:
            t0, t1 = t0, t_mid
            w0, w1 = w0, w_mid
            parent = seed_l
        else:
            t0, t1 = t_mid, t1
            w0, w1 = w_mid, w1
            parent = seed_r

        seed_v, seed_l, seed_r = parent.spawn(3)
        seed_v, = seed_v.generate_state(1)

        t_mid = (t0 + t1) / 2
        w_mid = utils.brownian_bridge(t0=t0, t1=t1, w0=w0, w1=w1, t=t_mid, seed=seed_v)
        depth += 1

    return w_mid, depth


def _create_cache(t0, t1, w0, w1, parent, k):
    ts = [t0, t1]
    ws = [w0, w1]

    seeds = [parent]

    for level in range(1, k + 1):
        new_ts = []
        new_ws = []
        new_seeds = []
        for i, parent in enumerate(seeds):
            seed_v, seed_l, seed_r = parent.spawn(3)
            seed_v, = seed_v.generate_state(1)
            new_seeds.extend([seed_l, seed_r])

            t0, t1 = ts[i], ts[i + 1]
            w0, w1 = ws[i], ws[i + 1]
            t = (t0 + t1) / 2
            w = utils.brownian_bridge(t0=t0, t1=t1, w0=w0, w1=w1, t=t, seed=seed_v)
            new_ts.extend([ts[i], t])
            new_ws.extend([ws[i], w])

        new_ts.append(ts[-1])
        new_ws.append(ws[-1])
        ts = new_ts
        ws = new_ws
        seeds = new_seeds

    # ts and ws have 2 ** k - 1 + 2 entries.
    # seeds have 2 ** k entries.
    return ts, ws, seeds

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
import warnings

import blist
import torch

from . import base_brownian
from . import utils
from .._core.misc import handle_unused_kwargs
from ..settings import LEVY_AREA_APPROXIMATIONS
from ..types import Scalar, TensorOrTensors, Optional, Tuple, Union, Tensor, Sequence


class BrownianPath(base_brownian.BaseBrownian):
    """Fast Brownian motion with all increments stored.

    Trades in memory for speed.

    To use:
    >>> bm = BrownianPath(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]])
    """

    def __init__(self,
                 t0: Scalar,
                 w0: Optional[Tensor] = None,
                 shape: Optional[Sequence[int]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 window_size: int = 8,
                 levy_area_approximation: str = LEVY_AREA_APPROXIMATIONS.none,
                 **unused_kwargs):
        """Initialize Brownian path.

        Args:
            t0 (float or Tensor): Initial time.
            w0 (sequence of Tensor, optional): Initial state.
                Defaults to a tensor of zeros with prescribed shape.
            shape (tuple of int, optional): The shape of each Brownian sample.
                The last dimension is treated as the channel dimension and
                any/all preceding dimensions are treated as batch dimensions.
            dtype (torch.dtype): The dtype of each Brownian sample.
                Defaults to the PyTorch default.
            device (torch.device): The device of each Brownian sample.
                Defaults to the current device.
            window_size (int): Size of the window around last query for local
                search.
            levy_area_approximation (str): Whether to also approximate Levy
                area. Defaults to None. Valid options are either 'none',
                'space-time', 'davie' or 'foster', corresponding to
                approximation type. This is needed for some higher-order SDE
                solvers.
        """
        # TODO: Couple of things to optimize: 1) search based on local window,
        #  2) avoid the `return_U` and `return_A` arguments.
        handle_unused_kwargs(unused_kwargs, msg=self.__class__.__name__)
        del unused_kwargs

        super(BrownianPath, self).__init__()
        if not utils.is_scalar(t0):
            raise ValueError('Initial time `t0` should be a float or 0-d torch.Tensor.')

        shape, dtype, device = utils.check_tensor_info(w0, shape=shape, dtype=dtype, device=device, name='`w0`')

        # Provide references so that point-based queries still work.
        t0 = float(t0)
        w0 = torch.zeros(size=shape, device=device, dtype=dtype) if w0 is None else w0
        self._t0 = t0
        self._w0 = w0
        self._ts = blist.blist([t0])  # noqa

        # Record the increments W_{s,t} and U_{s,t}.
        self._ws = blist.blist([torch.zeros_like(w0)])  # noqa
        self._us = blist.blist([torch.zeros_like(w0)])  # noqa

        self._last_idx = 0
        self._window_size = window_size

        # Avoid having if-statements in method body for speed.
        if levy_area_approximation == LEVY_AREA_APPROXIMATIONS.none:
            self._update_state = self._update_state_without_levy_area
            self._insert = self._insert_without_levy_area
        else:
            self._update_state = self._update_state_with_levy_area
            self._insert = self._insert_with_levy_area
        self.levy_area_approximation = levy_area_approximation

    def _update_state_without_levy_area(self, t: float) -> int:
        """Update the recorded Brownian state.

        Returns:
            The index of the old state or newly inserted state.
        """
        idx = bisect.bisect_left(self._ts, t)
        if idx < len(self._ts) and t == self._ts[idx]:
            return idx

        if idx <= 0:
            h1 = self._ts[0] - t
            W_h1 = utils.brownian_bridge_augmented(self._w0, h1)
            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
        elif idx >= len(self._ts):
            h1 = t - self._ts[-1]
            W_h1 = utils.brownian_bridge_augmented(self._w0, h1)
            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
        else:
            h = self._ts[idx] - self._ts[idx - 1]
            h1 = t - self._ts[idx - 1]

            W_h = self._ws[idx]
            W_h1 = utils.brownian_bridge_augmented(self._w0, h1, h, W_h)
            W_h2 = W_h - W_h1

            self._ws[idx] = W_h2
            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
        return idx

    def _update_state_with_levy_area(self, t: float) -> int:
        """Update the recorded Brownian state and 'shifted' space-time Levy area.

        Returns:
            The index of the old state or newly inserted state.
        """
        idx = bisect.bisect_left(self._ts, t)
        if idx < len(self._ts) and t == self._ts[idx]:
            return idx

        if idx <= 0:
            h1 = self._ts[0] - t
            W_h1, U_h1 = utils.brownian_bridge_augmented(
                self._w0, h1, levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time)
            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
            self._us.insert(idx, U_h1)
        elif idx >= len(self._ts):
            h1 = t - self._ts[-1]
            W_h1, U_h1 = utils.brownian_bridge_augmented(
                self._w0, h1, levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time)
            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
            self._us.insert(idx, U_h1)
        else:
            h = self._ts[idx] - self._ts[idx - 1]
            h1 = t - self._ts[idx - 1]
            h2 = h - h1

            W_h, U_h = self._ws[idx], self._us[idx]
            W_h1, U_h1 = utils.brownian_bridge_augmented(
                self._w0, h1, h, W_h, U_h, levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time)

            W_h2 = W_h - W_h1
            U_h2 = U_h - U_h1 - h2 * W_h1

            self._ws[idx] = W_h2
            self._us[idx] = U_h2

            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
            self._us.insert(idx, U_h1)
        return idx

    def _aggregate_W(self, idx_a: int, idx_b: int) -> torch.Tensor:
        """Aggregate Brownian increments."""
        if idx_b > idx_a:
            return sum(self._ws[idx_a + 1: idx_b + 1])
        return torch.zeros_like(self._w0)

    def _aggregate_W_U(self, idx_a: int, idx_b: int) -> Tuple[torch.Tensor, ...]:
        """Aggregate Brownian increments and space-time Levy area."""
        W, U = self._ws[idx_a + 1], self._us[idx_a + 1]
        for i in range(idx_a + 2, idx_b + 1):
            U = U + self._us[i] + (self._ts[i] - self._ts[i - 1]) * W
            W = W + self._ws[i]
        return W, U

    def _aggregate_W_U_A(self, idx_a: int, idx_b: int) -> Tuple[torch.Tensor, ...]:
        """Aggregate Brownian increments, space-time Levy area, and space-space Levy area."""
        W, U = self._aggregate_W_U(idx_a, idx_b)
        h = self._ts[idx_b] - self._ts[idx_a]
        H = utils.U_to_H(W, U, h)
        # TODO: Store and aggregate A; this would require a partial rewrite.
        A = utils.davie_foster_approximation(W, H, h, self.levy_area_approximation)
        return W, U, A

    def _point_eval(self, t: float, return_U=False, return_A=False) -> TensorOrTensors:
        idx_a, idx_b = 0, self._update_state(t)
        if return_U:
            if return_A:
                W, U, A = self._aggregate_W_U_A(idx_a, idx_b)
                return W + self._w0, U, A
            else:
                W, U = self._aggregate_W_U(idx_a, idx_b)
                return W + self._w0, U
        else:
            if return_A:
                W, U, A = self._aggregate_W_U_A(idx_a, idx_b)
                return W + self._w0, A
            else:
                return self._aggregate_W(idx_a, idx_b) + self._w0

    def _interval_eval(self, ta: float, tb: float, return_U=False, return_A=False) -> TensorOrTensors:
        if ta > tb:
            raise RuntimeError(f"Query times ta={ta:.3f} and tb={tb:.3f} must respect ta <= tb.")

        idx_a = self._update_state(ta)
        idx_b = self._update_state(tb)
        if return_U:
            if return_A:
                return self._aggregate_W_U_A(idx_a, idx_b)
            else:
                return self._aggregate_W_U(idx_a, idx_b)
        else:
            if return_A:
                W, U, A = self._aggregate_W_U_A(idx_a, idx_b)
                return W, A
            else:
                return self._aggregate_W(idx_a, idx_b)

    def __call__(self, ta, tb=None, return_U=False, return_A=False) -> TensorOrTensors:
        ta = float(ta)
        tb = float(tb) if tb is not None else tb
        if ta < self._t0:
            warnings.warn(f"Should have ta>=t0 but got ta={ta} and t0={self._t0}.")
            ta = self._t0

        if tb is None:
            return self._point_eval(ta)
        return self._interval_eval(ta, tb, return_U=return_U, return_A=return_A)

    def _insert_without_levy_area(self, t, w):
        """Insert time and Brownian motion increment.

        Silently replaces the original states if `t` is already in the list, and
        returns the old state. Otherwise returns `None` if `t` is not inside.

        The method should only be used for testing purposes.
        """
        t = float(t)
        idx = bisect.bisect_left(self._ts, t)
        if idx < len(self._ts) and t == self._ts[idx]:
            W = self._ws.pop(idx)
            self._ws.insert(idx, w)
            return W

        self._ts.insert(idx, t)
        self._ws.insert(idx, w)
        return None

    def _insert_with_levy_area(self, t, w, u):
        """Insert time, Brownian increment, and space-time Levy area increment.

        Silently replaces the original states if `t` is already in the list, and
        returns the old state. Otherwise returns `None` if `t` is not inside.

        The method should only be used for testing purposes.
        """
        t = float(t)
        idx = bisect.bisect_left(self._ts, t)
        if idx < len(self._ts) and t == self._ts[idx]:
            W, U = self._ws.pop(idx), self._us.pop(idx)
            self._ws.insert(idx, w)
            self._us.insert(idx, u)
            return W, U

        self._ts.insert(idx, t)
        self._ws.insert(idx, w)
        self._us.insert(idx, u)
        return None, None

    def __repr__(self):
        return f"{self.__class__.__name__}(t0={self._ts[0]:.3f}, t1={self._ts[-1]:.3f})"

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
        return {'ts': self._ts, 'ws': self._ws, 'us': self._us}

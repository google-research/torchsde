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
import bisect
import math
from typing import Optional, Tuple, Union

import blist
import torch

from . import base_brownian
from . import utils
from .._core.misc import handle_unused_kwargs
from ..settings import LEVY_AREA_APPROXIMATIONS
from ..types import Scalar, TensorOrTensors


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
                 w0: Optional[torch.Tensor] = None,
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 window_size: int = 8,
                 levy_area_approximation: str = LEVY_AREA_APPROXIMATIONS.none,
                 **unused_kwargs):
        """Initialize Brownian path.

        Args:
            t0 (float or Tensor): Initial time.
            w0 (sequence of Tensor, optional): Initial state.
            shape (tuple of int, optional): The shape of each Brownian sample.
                The last dimension is treated as the channel dimension and
                any/all preceding dimensions are treated as batch dimensions.
            dtype (torch.dtype): The dtype of each Brownian sample.
                Defaults to the PyTorch default.
            device (str or torch.device): The device of each Brownian sample.
                Defaults to the current device.
            window_size (int): Size of the window around last query for local
                search.
            levy_area_approximation (str): Whether to also approximate Levy
                area. Defaults to None. Valid options are either 'none',
                'space-time', 'davie' or 'foster', corresponding to
                approximation type. This is needed for some higher-order SDE
                solvers.
        """
        handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        super(BrownianPath, self).__init__()
        if not utils.is_scalar(t0):
            raise ValueError('Initial time `t0` should be a float or 0-d torch.Tensor.')
        t0 = float(t0)

        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")

        shapes = utils.get_tensors_info(w0, shape=True, default_shape=shape)
        dtypes = utils.get_tensors_info(w0, dtype=True, default_dtype=dtype)
        devices = utils.get_tensors_info(w0, device=True, default_device=device)
        if len(shapes) == 0:
            raise ValueError("Must either specify `shape` or pass in `w0` to implicitly define the shape.")
        if len(dtypes) == 0:
            raise ValueError("Must either specify `dtype` or pass in `w0` to implicitly define the dtype.")
        if len(devices) == 0:
            raise ValueError("Must either specify `device` or pass in `w0` to implicitly define the device.")

        # Make sure the reduce actually does a comparison, to get a bool datatype.
        shapes.append(shapes[-1])
        dtypes.append(dtypes[-1])
        devices.append(devices[-1])
        if not functools.reduce(operator.eq, shapes):
            raise ValueError("Multiple shapes found. Make sure `shape` and `w0` are consistent.")
        if not functools.reduce(operator.eq, dtypes):
            raise ValueError("Multiple dtypes found. Make sure `shape` and `w0` are consistent.")
        if not functools.reduce(operator.eq, devices):
            raise ValueError("Multiple devices found. Make sure `shape` and `w0` are consistent.")

        if w0 is None:
            w0 = torch.zeros(size=shape, device=device, dtype=dtype)

        if levy_area_approximation not in (LEVY_AREA_APPROXIMATIONS.none, LEVY_AREA_APPROXIMATIONS.space_time):
            raise ValueError(
                "BrownianPath currently only supports 'none' and 'space-time' for Levy area approximation."
            )
        self.levy_area_approximation = levy_area_approximation

        # Provide references so that point-based queries still work.
        self._t0 = t0
        self._w0 = w0
        self._ts = blist.blist([t0])
        self._ws = blist.blist([torch.zeros_like(w0)])  # Record W increment.
        self._us = blist.blist([torch.zeros_like(w0)])  # Record U increment.

        self._last_idx = 0
        self._window_size = window_size

    def _update_internal_state(self, t: float) -> int:
        """Update the recorded Brownian state and space-time Levy area.

        Returns:
            The index of the old state or newly inserted state.
        """
        idx = bisect.bisect_left(self._ts, t)
        if t == self._ts[idx]:
            return idx

        if idx <= 0:
            h1 = self._ts[0] - t
            W_h1, U_h1 = utils.brownian_bridge_augmented(self._w0, h1)
            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
            self._us.insert(idx, U_h1)
        elif idx >= len(self._ts):
            h1 = t - self._ts[-1]
            W_h1, U_h1 = utils.brownian_bridge_augmented(self._w0, h1)
            self._ts.insert(idx, t)
            self._ws.insert(idx, W_h1)
            self._us.insert(idx, U_h1)
        else:
            h = self._ts[idx] - self._ts[idx - 1]
            h1 = t - self._ts[idx - 1]
            h2 = h - h1

            W_h, U_h = self._ws[idx], self._us[idx]
            W_h1, U_h1 = utils.brownian_bridge_augmented(self._w0, h1, h, W_h, U_h)

            W_h2 = W_h - W_h1
            U_h2 = U_h - U_h1 - h2 * W_h1

            # Update right end.
            self._ws[idx] = W_h2
            self._us[idx] = U_h2

            # Insert new.
            self._ws.insert(idx, W_h1)
            self._us.insert(idx, U_h1)

        return idx

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
        """Insert time and Brownian motion state into lists.

        Silently replaces the original state if `t` is already in the list, and
        returns the old state. Otherwise returns `None` if `t` is not inside.

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
            idx = bisect.bisect_left(self._ts, t)
            if t == self._ts[idx]:
                old = self._ws[idx]
                self._ts.pop(idx)
                self._ws.pop(idx)
            else:
                old = None

        self._ts.insert(idx, t)
        self._ws.insert(idx, w)
        self._last_idx = idx
        return old

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

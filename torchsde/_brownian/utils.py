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
import math
from typing import Optional

import blist
import torch
from numpy.random import default_rng

from ..settings import LEVY_AREA_APPROXIMATIONS
from ..types import TensorOrTensors

_rsqrt3 = 1 / math.sqrt(3)


def search(ts: blist.blist, ws: blist.blist, t):
    """Search for the state that corresponds to the time.

    It is possible that `t` is not within the range of the sorted `ts`.

    Returns:
        (None, None, False) if `t` is not within the range of `ts`.
        (int, Tensor, True) if `t` is in `ts`.
        (int, Tensor, False) if `t` is not in `ts` but within the range.
    """
    if t == ts[-1]:  # Heuristic #1.
        idx = len(ts) - 1
        w = ws[idx]
        found = True
    elif len(ts) > 1 and t == ts[-2]:  # Heuristic #2.
        idx = len(ts) - 2
        w = ws[idx]
        found = True
    elif t == ts[0]:  # Heuristic #3.
        idx = 0
        w = ws[idx]
        found = True
    elif t > ts[-1] or t < ts[0]:  # `t` not within range.
        idx = None
        w = None
        found = False
    else:  # `t` within range.
        idx = bisect.bisect_left(ts, t)
        if t == ts[idx]:  # Found `t` in `ts`.
            w = ws[idx]
            found = True
        else:
            # Didn't find `t` in `ts`.
            t0, t1 = ts[idx - 1], ts[idx]
            w0, w1 = ws[idx - 1], ws[idx]
            w = brownian_bridge(t0=t0, t1=t1, w0=w0, w1=w1, t=t)
            found = False
    return idx, w, found


def search_and_insert(ts: blist.blist, ws: blist.blist, t):
    """Search for the state value that corresponds to the time; modify the lists if necessary."""
    # `t` has to already be in the range of `ts`.
    idx, w, found = search(ts=ts, ws=ws, t=t)
    if idx is not None and not found:
        ts.insert(idx, t)
        ws.insert(idx, w)
    return w


def normal_like(seed, ref):
    """Return a tensor sampled from standard Gaussian with shape that of `ref`.

    Randomness here is based on numpy!
    """
    if not isinstance(ref, torch.Tensor):
        raise ValueError(f'Reference should be a torch tensor, but is of type {type(ref)}.')
    return torch.tensor(default_rng(seed).normal(size=ref.shape)).to(ref)


def brownian_bridge(t0: float, t1: float, w0, w1, t: float, seed=None):
    # TODO: Use `torch.Generator`.
    mean = ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0)
    std = math.sqrt((t1 - t) * (t - t0) / (t1 - t0))
    if seed is not None:
        return mean + std * normal_like(seed, ref=mean)
    return mean + std * torch.randn_like(mean)


def augmented_brownian_bridge(
        s: float,
        ws: torch.Tensor,
        m: float,
        t: Optional[float] = None,
        wt: Optional[torch.Tensor] = None,
        ust: Optional[torch.Tensor] = None,
        levy_area_approximation: str = LEVY_AREA_APPROXIMATIONS.none) -> TensorOrTensors:
    if t is None:
        h = m - s
        wsm = math.sqrt(h) * torch.randn_like(ws)
        wm = ws + wsm
        if levy_area_approximation == LEVY_AREA_APPROXIMATIONS.none:
            return wm
        else:
            usm = h / 2. * (wsm + torch.randn_like(ws) * math.sqrt(h) * _rsqrt3)
            return wm, usm

    # Small operations performed on CPU.
    h1, h2, h = m - s, t - m, t - s
    A = torch.tensor(
        [[h1, h1 ** 2 / 2],
         [h1 ** 2 / 2, h1 ** 3 / 3]]
    )
    B = torch.tensor(
        [[h, h ** 2 / 2],
         [h ** 2 / 2, h ** 3 / 3]]
    )
    C = torch.tensor(
        [[h1, h1 ** 2 / 2 + h1 * h2],
         [h1 ** 2 / 2, h1 ** 3 / 3 + h1 ** 2 * h2 / 2]]
    )

    mu_x = torch.stack((ws, torch.zeros_like(ws)), dim=-1)
    mu_y = torch.stack((ws, torch.zeros_like(ws)), dim=-1)
    y = torch.stack((wt, ust), dim=-1)
    mean = mu_x + (y - mu_y) @ (C @ torch.inverse(B).to(ws)).T

    covariance = A - C @ torch.inverse(B) @ C.T
    L = torch.cholesky(covariance).to(ws)
    sample = mean + torch.randn_like(mean) @ L.T

    wm, usm = sample[..., 0], sample[..., 1]
    return [wm, usm]


def is_scalar(x):
    return isinstance(x, int) or isinstance(x, float) or (isinstance(x, torch.Tensor) and x.numel() == 1)


def blist_to(l, *args, **kwargs):  # noqa
    return blist.blist([li.to(*args, **kwargs) for li in l])  # noqa


def space_time_levy_area(W, h, levy_area_approximation, get_noise):
    if levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.space_time,
                                   LEVY_AREA_APPROXIMATIONS.davie,
                                   LEVY_AREA_APPROXIMATIONS.foster):
        return h / 2. * (W + get_noise() * math.sqrt(h) * _rsqrt3)
    else:
        return None


def davie_foster_approximation(W, H, h, levy_area_approximation, get_noise):
    if levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.none, LEVY_AREA_APPROXIMATIONS.space_time):
        return None
    elif W.shape == ():
        return torch.zeros_like(W)
    else:
        # Davie's approximation to the Levy area from space-time Levy area
        A = H.unsqueeze(-1) * W.unsqueeze(-2) - W.unsqueeze(-1) * H.unsqueeze(-2)
        if levy_area_approximation == LEVY_AREA_APPROXIMATIONS.foster:
            # Foster's additional correction to Davie's approximation
            tenth_h = 0.1 * h
            H_squared = H ** 2
            var = tenth_h * (tenth_h + H_squared.unsqueeze(-1) + H_squared.unsqueeze(-2))
            noise = get_noise()
            noise = noise - noise.transpose(-1, -2)
            # noise is skew symmetric of variance 2
            a_tilde = math.sqrt(var) * noise
            A += a_tilde
        return A


def get_tensors_info(*args,
                     shape=False,
                     dtype=False,
                     device=False,
                     default_shape=None,
                     default_dtype=None,
                     default_device=None):
    # Assume one of `shape`, `dtype`, and `device` should be true.
    if shape:
        shapes = [] if default_shape is None else [default_shape]
        return shapes + [arg.shape for arg in args if torch.is_tensor(arg)]
    if dtype:
        dtypes = [] if default_dtype is None else [default_dtype]
        return dtypes + [arg.dtype for arg in args if torch.is_tensor(arg)]
    if device:
        devices = [] if default_device is None else [default_device]
        return devices + [arg.device for arg in args if torch.is_tensor(arg)]

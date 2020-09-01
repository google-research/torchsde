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

"""Compare the speed of 5 Brownian motion variants on problems of different sizes."""
import argparse
import logging
import os
import time

import numpy.random as npr
import torch

import torchsde
from diagnostics import utils

t0, t1 = 0.0, 1.0
reps, steps = 3, 100
small_batch_size, small_d = 128, 5
large_batch_size, large_d = 256, 128
huge_batch_size, huge_d = 512, 256


def _time_query(bm, ts):
    now = time.perf_counter()
    for _ in range(reps):
        for ta, tb in zip(ts[:-1], ts[1:]):
            if ta > tb:
                ta, tb = tb, ta
            bm(ta, tb)
    return time.perf_counter() - now


def _compare(w0, ts, msg=''):
    bm = torchsde.brownian_lib.BrownianPath(t0=t0, w0=w0)
    bp_cpp_time = _time_query(bm, ts)
    logging.warning(f'{msg} (brownian_lib.BrownianPath): {bp_cpp_time:.4f}')

    bm = torchsde.BrownianPath(t0=t0, w0=w0)
    bp_py_time = _time_query(bm, ts)
    logging.warning(f'{msg} (torchsde.BrownianPath): {bp_py_time:.4f}')

    bm = torchsde.brownian_lib.BrownianTree(t0=t0, w0=w0, t1=t1, tol=1e-5)
    bt_cpp_time = _time_query(bm, ts)
    logging.warning(f'{msg} (brownian_lib.BrownianTree): {bt_cpp_time:.4f}')

    bm = torchsde.BrownianTree(t0=t0, w0=w0, t1=t1, tol=1e-5)
    bt_py_time = _time_query(bm, ts)
    logging.warning(f'{msg} (torchsde.BrownianTree): {bt_py_time:.4f}')

    bm = torchsde.BrownianInterval(t0=t0, t1=t1, shape=w0.shape, dtype=w0.dtype, device=w0.device)
    bi_py_time = _time_query(bm, ts)
    logging.warning(f'{msg} (torchsde.BrownianInterval): {bi_py_time:.4f}')

    return bp_cpp_time, bp_py_time, bt_cpp_time, bt_py_time, bi_py_time


def sequential_access():
    ts = torch.linspace(t0, t1, steps=steps)

    w0 = torch.zeros(small_batch_size, small_d).to(device)
    bp_cpp_time_s, bp_py_time_s, bt_cpp_time_s, bt_py_time_s, bi_py_time_s = _compare(w0, ts,
                                                                                      msg='small sequential access')

    w0 = torch.zeros(large_batch_size, large_d).to(device)
    bp_cpp_time_l, bp_py_time_l, bt_cpp_time_l, bt_py_time_l, bi_py_time_l = _compare(w0, ts,
                                                                                      msg='large sequential access')

    w0 = torch.zeros(huge_batch_size, huge_d).to(device)
    bp_cpp_time_h, bp_py_time_h, bt_cpp_time_h, bt_py_time_h, bi_py_time_h = _compare(w0, ts,
                                                                                      msg="huge sequential access")

    img_path = os.path.join('.', 'benchmarks', f'plots-{device}', 'sequential_access.png')
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    xaxis = [small_batch_size * small_d, large_batch_size * large_batch_size, huge_batch_size * huge_d]

    utils.swiss_knife_plotter(
        img_path,
        plots=[
            {'x': xaxis, 'y': [bp_cpp_time_s, bp_cpp_time_l, bp_cpp_time_h], 'label': 'bp_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bp_py_time_s, bp_py_time_l, bp_py_time_h], 'label': 'bp_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_cpp_time_s, bt_cpp_time_l, bt_cpp_time_h], 'label': 'bt_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_py_time_s, bt_py_time_l, bt_py_time_h], 'label': 'bt_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bi_py_time_s, bi_py_time_l, bi_py_time_h], 'label': 'bi_py', 'marker': 'x'},
        ],
        options={
            'xscale': 'log',
            'yscale': 'log',
            'xlabel': 'size of tensor',
            'ylabel': f'wall time on {device}',
            'title': 'sequential access'
        }
    )


def random_access():
    ts = torch.empty(steps).uniform_(t0, t1)

    w0 = torch.zeros(small_batch_size, small_d).to(device)
    bp_cpp_time_s, bp_py_time_s, bt_cpp_time_s, bt_py_time_s, bi_py_time_s = _compare(w0, ts, msg='small random access')

    w0 = torch.zeros(large_batch_size, large_d).to(device)
    bp_cpp_time_l, bp_py_time_l, bt_cpp_time_l, bt_py_time_l, bi_py_time_l = _compare(w0, ts, msg='large random access')

    w0 = torch.zeros(huge_batch_size, huge_d).to(device)
    bp_cpp_time_h, bp_py_time_h, bt_cpp_time_h, bt_py_time_h, bi_py_time_h = _compare(w0, ts, msg="huge random access")

    img_path = os.path.join('.', 'benchmarks', f'plots-{device}', 'random_access.png')
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    xaxis = [small_batch_size * small_d, large_batch_size * large_batch_size, huge_batch_size * huge_d]

    utils.swiss_knife_plotter(
        img_path,
        plots=[
            {'x': xaxis, 'y': [bp_cpp_time_s, bp_cpp_time_l, bp_cpp_time_h], 'label': 'bp_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bp_py_time_s, bp_py_time_l, bp_py_time_h], 'label': 'bp_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_cpp_time_s, bt_cpp_time_l, bt_cpp_time_h], 'label': 'bt_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_py_time_s, bt_py_time_l, bt_py_time_h], 'label': 'bt_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bi_py_time_s, bi_py_time_l, bi_py_time_h], 'label': 'bi_py', 'marker': 'x'},
        ],
        options={
            'xscale': 'log',
            'yscale': 'log',
            'xlabel': 'size of tensor',
            'ylabel': f'wall time on {device}',
            'title': 'random access'
        }
    )


class SDE(torchsde.SDEIto):
    def __init__(self):
        super(SDE, self).__init__(noise_type="diagonal")

    def f(self, t, y):
        return y

    def g(self, t, y):
        return torch.exp(-y)


def _time_sdeint(sde, y0, ts, bm):
    now = time.perf_counter()
    with torch.no_grad():
        torchsde.sdeint(sde, y0, ts, bm, method='euler')
    return time.perf_counter() - now


def _time_sdeint_bp(sde, y0, ts, bm):
    now = time.perf_counter()
    sde.zero_grad()
    y0 = y0.clone().requires_grad_(True)
    ys = torchsde.sdeint(sde, y0, ts, bm, method='euler')
    ys.sum().backward()
    return time.perf_counter() - now


def _time_sdeint_adjoint(sde, y0, ts, bm):
    now = time.perf_counter()
    sde.zero_grad()
    y0 = y0.clone().requires_grad_(True)
    ys = torchsde.sdeint_adjoint(sde, y0, ts, bm, method='euler')
    ys.sum().backward()
    return time.perf_counter() - now


def _compare_sdeint(w0, sde, y0, ts, func, msg=''):
    bm = torchsde.brownian_lib.BrownianPath(t0, w0)
    bp_cpp_time = func(sde, y0, ts, bm)
    logging.warning(f'{msg} (brownian_lib.BrownianPath): {bp_cpp_time:.4f}')

    bm = torchsde.BrownianPath(t0, w0)
    bp_py_time = func(sde, y0, ts, bm)
    logging.warning(f'{msg} (torchsde.BrownianPath): {bp_py_time:.4f}')

    bm = torchsde.brownian_lib.BrownianTree(t0, w0)
    bt_cpp_time = func(sde, y0, ts, bm)
    logging.warning(f'{msg} (brownian_lib.BrownianTree): {bt_cpp_time:.4f}')

    bm = torchsde.BrownianTree(t0, w0)
    bt_py_time = func(sde, y0, ts, bm)
    logging.warning(f'{msg} (torchsde.BrownianTree): {bt_py_time:.4f}')

    bm = torchsde.BrownianInterval(t0, t0 + 1, shape=w0.shape, dtype=w0.dtype, device=w0.device)
    bi_py_time = func(sde, y0, ts, bm)
    logging.warning(f'{msg} (torchsde.BrownianInterval): {bi_py_time:.4f}')

    return bp_cpp_time, bp_py_time, bt_cpp_time, bt_py_time, bi_py_time


def solver_access(func=_time_sdeint):
    ts = torch.linspace(t0, t1, steps)
    sde = SDE().to(device)

    y0 = w0 = torch.zeros(small_batch_size, small_d).to(device)
    bp_cpp_time_s, bp_py_time_s, bt_cpp_time_s, bt_py_time_s, bi_py_time_s = _compare_sdeint(w0, sde, y0, ts, func,
                                                                                             msg='small')

    y0 = w0 = torch.zeros(large_batch_size, large_d).to(device)
    bp_cpp_time_l, bp_py_time_l, bt_cpp_time_l, bt_py_time_l, bi_py_time_l = _compare_sdeint(w0, sde, y0, ts, func,
                                                                                             msg='large')

    y0 = w0 = torch.zeros(huge_batch_size, huge_d).to(device)
    bp_cpp_time_h, bp_py_time_h, bt_cpp_time_h, bt_py_time_h, bi_py_time_h = _compare_sdeint(w0, sde, y0, ts, func,
                                                                                             msg='huge')

    name = {
        _time_sdeint: 'sdeint',
        _time_sdeint_bp: 'sdeint-backprop-solver',
        _time_sdeint_adjoint: 'sdeint-backprop-adjoint'
    }[func]

    img_path = os.path.join('.', 'benchmarks', f'plots-{device}', f'{name}.png')
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    xaxis = [small_batch_size * small_d, large_batch_size * large_batch_size, huge_batch_size * huge_d]

    utils.swiss_knife_plotter(
        img_path,
        plots=[
            {'x': xaxis, 'y': [bp_cpp_time_s, bp_cpp_time_l, bp_cpp_time_h], 'label': 'bp_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bp_py_time_s, bp_py_time_l, bp_py_time_h], 'label': 'bp_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_cpp_time_s, bt_cpp_time_l, bt_cpp_time_h], 'label': 'bt_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_py_time_s, bt_py_time_l, bt_py_time_h], 'label': 'bt_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bi_py_time_s, bi_py_time_l, bi_py_time_h], 'label': 'bi_py', 'marker': 'x'},
        ],
        options={
            'xscale': 'log',
            'yscale': 'log',
            'xlabel': 'size of tensor',
            'ylabel': f'wall time on {device}',
            'title': name
        }
    )


def main():
    sequential_access()
    random_access()

    solver_access(func=_time_sdeint)
    solver_access(func=_time_sdeint_bp)
    solver_access(func=_time_sdeint_adjoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')

    npr.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    main()

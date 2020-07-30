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

"""Compare the speed of 4 Brownian motion variants on problems of different size."""
import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy.random as npr
import torch

import torchsde
import torchsde.brownian_lib as brownian_lib

t0, t1 = 0.0, 1.0
reps, steps = 3, 100
small_batch_size, small_d = 128, 5
large_batch_size, large_d = 256, 128
huge_batch_size, huge_d = 512, 256


def swiss_knife_plotter(img_path, plots=None, scatters=None, options=None):
    if plots is None: plots = ()
    if scatters is None: scatters = ()
    if options is None: options = {}

    plt.figure()
    if 'xscale' in options: plt.xscale(options['xscale'])
    if 'yscale' in options: plt.yscale(options['yscale'])
    if 'xlabel' in options: plt.xlabel(options['xlabel'])
    if 'ylabel' in options: plt.ylabel(options['ylabel'])
    if 'title' in options: plt.title(options['title'])

    for entry in plots:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        plt.plot(entry['x'], entry['y'], **kwargs)
    for entry in scatters:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        plt.scatter(entry['x'], entry['y'], **kwargs)

    if len(plots) > 0 or len(scatters) > 0: plt.legend()
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def _time_query(bm, ts):
    now = time.perf_counter()
    for _ in range(reps):
        for t in ts:
            bm(t)
    return time.perf_counter() - now


def _compare(w0, ts, msg=''):
    bm = brownian_lib.BrownianPath(t0=t0, w0=w0)
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

    return bp_cpp_time, bp_py_time, bt_cpp_time, bt_py_time


def sequential_access():
    ts = torch.linspace(t0, t1, steps=steps)

    w0 = torch.zeros(small_batch_size, small_d).to(device)
    bp_cpp_time_s, bp_py_time_s, bt_cpp_time_s, bt_py_time_s = _compare(w0, ts, msg='small sequential access')

    w0 = torch.zeros(large_batch_size, large_d).to(device)
    bp_cpp_time_l, bp_py_time_l, bt_cpp_time_l, bt_py_time_l = _compare(w0, ts, msg='large sequential access')

    w0 = torch.zeros(huge_batch_size, huge_d).to(device)
    bp_cpp_time_h, bp_py_time_h, bt_cpp_time_h, bt_py_time_h = _compare(w0, ts, msg="huge sequential access")

    img_path = os.path.join('.', 'benchmarks', 'plots', 'sequential_access.png')
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    xaxis = [small_batch_size * small_d, large_batch_size * large_batch_size, huge_batch_size * huge_d]

    swiss_knife_plotter(
        img_path,
        plots=[
            {'x': xaxis, 'y': [bp_cpp_time_s, bp_cpp_time_l, bp_cpp_time_h], 'label': 'bp_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bp_py_time_s, bp_py_time_l, bp_py_time_h], 'label': 'bp_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_cpp_time_s, bt_cpp_time_l, bt_cpp_time_h], 'label': 'bt_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_py_time_s, bt_py_time_l, bt_py_time_h], 'label': 'bt_py', 'marker': 'x'},
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
    bp_cpp_time_s, bp_py_time_s, bt_cpp_time_s, bt_py_time_s = _compare(w0, ts, msg='small random access')

    w0 = torch.zeros(large_batch_size, large_d).to(device)
    bp_cpp_time_l, bp_py_time_l, bt_cpp_time_l, bt_py_time_l = _compare(w0, ts, msg='large random access')

    w0 = torch.zeros(huge_batch_size, huge_d).to(device)
    bp_cpp_time_h, bp_py_time_h, bt_cpp_time_h, bt_py_time_h = _compare(w0, ts, msg="huge random access")

    img_path = os.path.join('.', 'benchmarks', 'plots', 'random_access.png')
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    xaxis = [small_batch_size * small_d, large_batch_size * large_batch_size, huge_batch_size * huge_d]

    swiss_knife_plotter(
        img_path,
        plots=[
            {'x': xaxis, 'y': [bp_cpp_time_s, bp_cpp_time_l, bp_cpp_time_h], 'label': 'bp_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bp_py_time_s, bp_py_time_l, bp_py_time_h], 'label': 'bp_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_cpp_time_s, bt_cpp_time_l, bt_cpp_time_h], 'label': 'bt_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_py_time_s, bt_py_time_l, bt_py_time_h], 'label': 'bt_py', 'marker': 'x'},
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

    return bp_cpp_time, bp_py_time, bt_cpp_time, bt_py_time


def solver_access(func=_time_sdeint):
    ts = torch.linspace(t0, t1, steps)
    sde = SDE().to(device)

    y0 = w0 = torch.zeros(small_batch_size, small_d).to(device)
    bp_cpp_time_s, bp_py_time_s, bt_cpp_time_s, bt_py_time_s = _compare_sdeint(w0, sde, y0, ts, func, msg='small')

    y0 = w0 = torch.zeros(large_batch_size, large_d).to(device)
    bp_cpp_time_l, bp_py_time_l, bt_cpp_time_l, bt_py_time_l = _compare_sdeint(w0, sde, y0, ts, func, msg='large')

    y0 = w0 = torch.zeros(huge_batch_size, huge_d).to(device)
    bp_cpp_time_h, bp_py_time_h, bt_cpp_time_h, bt_py_time_h = _compare_sdeint(w0, sde, y0, ts, func, msg='huge')

    name = {
        _time_sdeint: 'sdeint',
        _time_sdeint_bp: 'sdeint-backprop-solver',
        _time_sdeint_adjoint: 'sdeint-backprop-adjoint'
    }[func]

    img_path = os.path.join('.', 'benchmarks', 'plots', f'{name}.png')
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    xaxis = [small_batch_size * small_d, large_batch_size * large_batch_size, huge_batch_size * huge_d]

    swiss_knife_plotter(
        img_path,
        plots=[
            {'x': xaxis, 'y': [bp_cpp_time_s, bp_cpp_time_l, bp_cpp_time_h], 'label': 'bp_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bp_py_time_s, bp_py_time_l, bp_py_time_h], 'label': 'bp_py', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_cpp_time_s, bt_cpp_time_l, bt_cpp_time_h], 'label': 'bt_cpp', 'marker': 'x'},
            {'x': xaxis, 'y': [bt_py_time_s, bt_py_time_l, bt_py_time_h], 'label': 'bt_py', 'marker': 'x'},
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

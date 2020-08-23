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

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from scipy import stats

from tests.basic_sde import AdditiveSDE
from tests.problems import Ex3Additive
from torchsde import sdeint, BrownianInterval
from torchsde.settings import LEVY_AREA_APPROXIMATIONS
from .utils import to_numpy, makedirs_if_not_found, compute_mse


def inspect_samples():
    batch_size, d, m = 32, 1, 5
    steps = 10

    ts = torch.linspace(0., 5., steps=steps).to(device)
    dt = 3e-1
    y0 = torch.ones(batch_size, d).to(device)
    sde = AdditiveSDE(d=d, m=m).to(device)

    with torch.no_grad():
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], shape=(batch_size, m), dtype=y0.dtype, device=device,
                              levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time)
        ys_em = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='euler')
        ys_srk = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='srk')
        ys_true = sdeint(sde, y0=y0, ts=ts, dt=1e-3, bm=bm, method='euler')

        ys_em = ys_em.squeeze().t()
        ys_srk = ys_srk.squeeze().t()
        ys_true = ys_true.squeeze().t()

        ts_, ys_em_, ys_srk_, ys_true_ = to_numpy(ts, ys_em, ys_srk, ys_true)

    # Visualize sample path.
    img_dir = os.path.join('.', 'diagnostics', 'plots', 'ito_additive')
    makedirs_if_not_found(img_dir)

    for i, (ys_em_i, ys_srk_i, ys_true_i) in enumerate(zip(ys_em_, ys_srk_, ys_true_)):
        plt.figure()
        plt.plot(ts_, ys_em_i, label='em')
        plt.plot(ts_, ys_srk_i, label='srk')
        plt.plot(ts_, ys_true_i, label='true')
        plt.legend()
        plt.savefig(os.path.join(img_dir, f'{i}'))
        plt.close()


def inspect_strong_order():
    batch_size, d, m = 4096, 5, 5
    ts = torch.tensor([0., 5.]).to(device)
    dts = tuple(2 ** -i for i in range(1, 9))
    y0 = torch.ones(batch_size, d).to(device)
    sde = Ex3Additive(d=d).to(device)

    euler_mses_ = []
    srk_mses_ = []

    with torch.no_grad():
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], shape=(batch_size, m), dtype=y0.dtype, device=device,
                              levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time)

        for dt in tqdm.tqdm(dts):
            # Only take end value.
            _, ys_euler = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='euler')
            _, ys_srk = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='srk')
            _, ys_analytical = sde.analytical_sample(y0=y0, ts=ts, bm=bm)

            euler_mse = compute_mse(ys_euler, ys_analytical)
            srk_mse = compute_mse(ys_srk, ys_analytical)

            euler_mse_, srk_mse_ = to_numpy(euler_mse, srk_mse)

            euler_mses_.append(euler_mse_)
            srk_mses_.append(srk_mse_)
    del euler_mse_, srk_mse_

    # Divide the log-error by 2, since textbook strong orders are represented so.
    log = lambda x: np.log(np.array(x))
    euler_slope, _, _, _, _ = stats.linregress(log(dts), log(euler_mses_) / 2)
    srk_slope, _, _, _, _ = stats.linregress(log(dts), log(srk_mses_) / 2)

    plt.figure()
    plt.plot(dts, euler_mses_, label=f'euler(k={euler_slope:.4f})')
    plt.plot(dts, srk_mses_, label=f'srk(k={srk_slope:.4f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    img_dir = os.path.join('.', 'diagnostics', 'plots', 'ito_additive')
    makedirs_if_not_found(img_dir)
    plt.savefig(os.path.join(img_dir, 'rate'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', action='store_true')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1147481649)

    inspect_samples()
    inspect_strong_order()

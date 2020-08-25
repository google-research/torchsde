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

from tests.problems import Ex1Scalar
from torchsde import sdeint, BrownianInterval
from torchsde.settings import LEVY_AREA_APPROXIMATIONS
from .utils import to_numpy, makedirs_if_not_found, compute_mse


def inspect_sample():
    batch_size, d = 32, 1
    steps = 100

    ts = torch.linspace(0., 5., steps=steps, device=device)
    dt = 1e-1
    y0 = torch.ones(batch_size, d, device=device)
    sde = Ex1Scalar(d=d, sde_type='stratonovich').to(device)
    sde.noise_type = "scalar"

    with torch.no_grad():
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], shape=(batch_size, 1), dtype=y0.dtype, device=device,
                              levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time)

        ys_heun = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='heun', names={'drift': 'f_corr'})
        ys_euler_heun = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='euler_heun', names={'drift': 'f_corr'})
        ys_midpoint = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='midpoint', names={'drift': 'f_corr'})
        ys_milstein_strat = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='milstein', names={'drift': 'f_corr'})
        ys_mil_strat_grad_free = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='milstein', names={'drift': 'f_corr'},
                                        options={'grad_free': True})
        ys_analytical = sde.analytical_sample(y0=y0, ts=ts, bm=bm)

        ys_heun = ys_heun.squeeze().t()
        ys_euler_heun = ys_euler_heun.squeeze().t()
        ys_midpoint = ys_midpoint.squeeze().t()
        ys_milstein_strat = ys_milstein_strat.squeeze().t()
        ys_mil_strat_grad_free = ys_mil_strat_grad_free.squeeze().t()
        ys_analytical = ys_analytical.squeeze().t()

        ts_, ys_heun_, ys_euler_heun_, ys_midpoint_, ys_milstein_strat_, ys_mil_strat_grad_free_, ys_analytical_ = to_numpy(
            ts, ys_heun, ys_euler_heun, ys_midpoint, ys_milstein_strat, ys_mil_strat_grad_free, ys_analytical)

    # Visualize sample path.
    img_dir = os.path.join('.', 'diagnostics', 'plots', 'stratonovich_scalar')
    makedirs_if_not_found(img_dir)

    for i, (ys_heun_i, ys_euler_heun_i, ys_midpoint_i, ys_milstein_strat_i, ys_mil_strat_grad_free_i, ys_analytical_i) in enumerate(
            zip(ys_heun_, ys_euler_heun_, ys_midpoint_, ys_milstein_strat_, ys_mil_strat_grad_free_, ys_analytical_)):
        plt.figure()
        plt.plot(ts_, ys_heun_i, label='heun')
        plt.plot(ts_, ys_euler_heun_i, label='euler_heun')
        plt.plot(ts_, ys_midpoint_i, label='midpoint')
        plt.plot(ts_, ys_milstein_strat_i, label='milstein_strat')
        plt.plot(ts_, ys_mil_strat_grad_free_i, label='milstein_strat_grad_free')
        plt.plot(ts_, ys_analytical_i, label='analytical')
        plt.legend()
        plt.savefig(os.path.join(img_dir, f'{i}'))
        plt.close()


def inspect_strong_order():
    batch_size, d = 4096, 10
    ts = torch.tensor([0., 5.], device=device)
    dts = tuple(2 ** -i for i in range(1, 9))
    y0 = torch.ones(batch_size, d, device=device)
    sde = Ex1Scalar(d=d, sde_type='stratonovich').to(device)
    sde.noise_type = "scalar"

    heun_mses_ = []
    euler_heun_mses_ = []
    midpoint_mses_ = []
    milstein_strat_mses_ = []
    mil_strat_grad_free_mses_ = []

    with torch.no_grad():
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], shape=(batch_size, 1), dtype=y0.dtype, device=device,
                              levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time)

        for dt in tqdm.tqdm(dts):
            # Only take end value.
            _, ys_heun = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='heun', names={'drift': 'f_corr'})
            _, ys_euler_heun = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='euler_heun', names={'drift': 'f_corr'})
            _, ys_midpoint = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='midpoint', names={'drift': 'f_corr'})
            _, ys_milstein_strat = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='milstein', names={'drift': 'f_corr'})
            _, ys_mil_strat_grad_free = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method='milstein',
                                               names={'drift': 'f_corr'}, options={'grad_free': True})
            _, ys_analytical = sde.analytical_sample(y0=y0, ts=ts, bm=bm)

            heun_mse = compute_mse(ys_heun, ys_analytical)
            euler_heun_mse = compute_mse(ys_euler_heun, ys_analytical)
            midpoint_mse = compute_mse(ys_midpoint, ys_analytical)
            milstein_strat_mse = compute_mse(ys_milstein_strat, ys_analytical)
            mil_strat_grad_free_mse = compute_mse(ys_mil_strat_grad_free, ys_analytical)

            heun_mse_, euler_heun_mse_, midpoint_mse_, milstein_strat_mse_, mil_strat_grad_free_mse_ = to_numpy(
                heun_mse, 
                euler_heun_mse, 
                midpoint_mse,
                milstein_strat_mse,
                mil_strat_grad_free_mse
            )

            heun_mses_.append(heun_mse_)
            euler_heun_mses_.append(euler_heun_mse_)
            midpoint_mses_.append(midpoint_mse_)
            milstein_strat_mses_.append(milstein_strat_mse_)
            mil_strat_grad_free_mses_.append(mil_strat_grad_free_mse_)
    del heun_mse_, euler_heun_mse_, midpoint_mse_, milstein_strat_mse_, mil_strat_grad_free_mse_

    # Divide the log-error by 2, since textbook strong orders are represented so.
    log = lambda x: np.log(np.array(x))
    heun_slope, _, _, _, _ = stats.linregress(log(dts), log(heun_mses_) / 2)
    euler_heun_slope, _, _, _, _ = stats.linregress(log(dts), log(euler_heun_mses_) / 2)
    midpoint_slope, _, _, _, _ = stats.linregress(log(dts), log(midpoint_mses_) / 2)
    milstein_strat_slope, _, _, _, _ = stats.linregress(log(dts), log(milstein_strat_mses_) / 2)
    mil_strat_grad_free_slope, _, _, _, _ = stats.linregress(log(dts), log(mil_strat_grad_free_mses_) / 2)

    plt.figure()
    plt.plot(dts, heun_mses_, label=f'heun(k={heun_slope:.4f})')
    plt.plot(dts, euler_heun_mses_, label=f'euler_heun(k={euler_heun_slope:.4f})')
    plt.plot(dts, midpoint_mses_, label=f'midpoint(k={midpoint_slope:.4f})')
    plt.plot(dts, milstein_strat_mses_, label=f'milstein_strat(k={milstein_strat_slope:.4f})')
    plt.plot(dts, mil_strat_grad_free_mses_, label=f'milstein_strat_grad_free(k={mil_strat_grad_free_slope:.4f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    img_dir = os.path.join('.', 'diagnostics', 'plots', 'stratonovich_scalar')
    makedirs_if_not_found(img_dir)
    plt.savefig(os.path.join(img_dir, 'rate'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', action='store_true')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    inspect_sample()
    inspect_strong_order()

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

import os

import torch
import tqdm

from tests.problems import Ex4
from torchsde import sdeint, BrownianInterval
from torchsde.settings import LEVY_AREA_APPROXIMATIONS
from . import utils


def inspect_sample():
    batch_size, d, m = 32, 1, 5
    steps = 100
    ts = torch.linspace(0., 5., steps=steps, device=device)
    dt = 1e-1
    y0 = torch.full((batch_size, d), fill_value=0.1, device=device)
    sde = Ex4(d=d, m=m).to(device)
    bm = BrownianInterval(t0=ts[0], t1=ts[-1], shape=(batch_size, m), dtype=y0.dtype, device=device,
                          levy_area_approximation=LEVY_AREA_APPROXIMATIONS.foster)

    with torch.no_grad():
        true = sdeint(sde, y0=y0, ts=ts, dt=2 ** -13, bm=bm, method="midpoint")

        midpoint = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method="midpoint")
        log_ode_midpoint = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method="log_ode_midpoint")

        true, midpoint, log_ode_midpoint = tuple(map(lambda x: x.squeeze().t(), (true, midpoint, log_ode_midpoint)))
        ts, true, midpoint, log_ode_midpoint = utils.to_numpy(ts, true, midpoint, log_ode_midpoint)

    img_dir = os.path.join('.', 'diagnostics', 'plots', 'stratonovich_general')
    utils.makedirs(img_dir)

    for i, (midpoint_i, log_ode_midpoint_i, true_i) in enumerate(zip(midpoint, log_ode_midpoint, true)):
        utils.swiss_knife_plotter(
            img_path=os.path.join(img_dir, f'{i}'),
            plots=[
                {'x': ts, 'y': midpoint_i, 'label': 'midpoint'},
                {'x': ts, 'y': log_ode_midpoint_i, 'label': 'log_ode_midpoint'},
                {'x': ts, 'y': true_i, 'label': 'true'},
            ]
        )


def inspect_strong_order():
    batch_size, d, m = 8192, 3, 5
    ts = torch.tensor([0., 5.], device=device)
    dts = tuple(2 ** -i for i in range(1, 9))
    y0 = torch.full((batch_size, d), fill_value=0.1, device=device)
    sde = Ex4(d=d, m=m).to(device)
    bm = BrownianInterval(t0=ts[0], t1=ts[-1], shape=(batch_size, m), dtype=y0.dtype, device=device,
                          levy_area_approximation=LEVY_AREA_APPROXIMATIONS.davie)
    _, true = sdeint(sde, y0=y0, ts=ts, dt=2 ** -13, bm=bm, method="midpoint")

    midpoint_mses = []
    log_ode_midpoint_mses = []

    with torch.no_grad():
        for dt in tqdm.tqdm(dts):
            _, midpoint = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method="midpoint")
            _, log_ode_midpoint = sdeint(sde, y0=y0, ts=ts, dt=dt, bm=bm, method="log_ode_midpoint")

            midpoint_mse = utils.compute_mse(midpoint, true)
            log_ode_midpoint_mse = utils.compute_mse(log_ode_midpoint, true)

            midpoint_mses.append(midpoint_mse)
            log_ode_midpoint_mses.append(log_ode_midpoint_mse)

    midpoint_slope = utils.regress_slope(utils.log(dts), utils.half_log(midpoint_mses))
    log_ode_midpoint_slope = utils.regress_slope(utils.log(dts), utils.half_log(log_ode_midpoint_mses))

    img_dir = os.path.join('.', 'diagnostics', 'plots', 'stratonovich_general')
    utils.makedirs(img_dir)

    utils.swiss_knife_plotter(
        img_path=os.path.join(img_dir, 'rate'),
        plots=[
            {'x': dts, 'y': midpoint_mses, 'label': f'midpoint(k={midpoint_slope:.4f})'},
            {'x': dts, 'y': log_ode_midpoint_mses, 'label': f'log_ode_midpoint(k={log_ode_midpoint_slope:.4f})'}
        ],
        options={'xscale': 'log', 'yscale': 'log'}
    )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1147481649)

    inspect_sample()
    inspect_strong_order()

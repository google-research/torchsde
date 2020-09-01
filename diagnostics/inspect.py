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

import copy
import os

import torch
import tqdm

from torchsde import BaseBrownian, BaseSDE, sdeint
from torchsde.settings import SDE_TYPES
from torchsde.types import Tensor, Vector, Scalar, Tuple, Optional
from . import utils


def inspect_samples(y0: Tensor,
                    ts: Vector,
                    dt: Scalar,
                    sde: BaseSDE,
                    bm: BaseBrownian,
                    img_dir: str,
                    methods: Tuple[str, ...],
                    vis_dim=0,
                    dt_true: Optional[float] = 2 ** -12):
    sde = copy.deepcopy(sde).requires_grad_(False)

    solns = [sdeint(sde, y0, ts, bm, method=method, dt=dt) for method in methods]

    method_for_true = 'euler' if sde.sde_type == SDE_TYPES.ito else 'midpoint'
    true = sdeint(sde, y0, ts, bm, method=method_for_true, dt=dt_true)
    methods += ('true',)
    solns += [true]

    # (T, batch_size, d) -> (T, batch_size) -> (batch_size, T).
    solns = [soln[..., vis_dim].t() for soln in solns]

    for i, samples in enumerate(zip(*solns)):
        utils.swiss_knife_plotter(
            img_path=os.path.join(img_dir, f'{i}'),
            plots=[
                {'x': ts, 'y': sample, 'label': method}
                for sample, method in zip(samples, methods)
            ]
        )


def inspect_strong_order(y0: Tensor,
                         t0: Scalar,
                         t1: Scalar,
                         dts: Vector,
                         sde: BaseSDE,
                         bm: BaseBrownian,
                         img_dir: str,
                         methods: Tuple[str, ...],
                         dt_true: Optional[float] = 2 ** -12):
    sde = copy.deepcopy(sde).requires_grad_(False)

    ts = torch.tensor([t0, t1], device=y0.device)
    method_for_true = 'euler' if sde.sde_type == SDE_TYPES.ito else 'midpoint'
    true = sdeint(sde, y0, ts, bm, method=method_for_true, dt=dt_true)[-1]

    mses = []
    for dt in tqdm.tqdm(dts):
        solns = [sdeint(sde, y0, ts, bm, method=method, dt=dt)[-1] for method in methods]
        mses_for_dt = [utils.compute_mse(soln, true) for soln in solns]
        mses.append(mses_for_dt)

    slopes = [
        utils.regress_slope(utils.log(dts), utils.half_log(mses_for_method))
        for mses_for_method in zip(*mses)
    ]

    utils.swiss_knife_plotter(
        img_path=os.path.join(img_dir, 'rate'),
        plots=[
            {'x': dts, 'y': mses_for_method, 'label': f'{method}(k={slope:.4f})'}
            for mses_for_method, method, slope in zip(zip(*mses), methods, slopes)
        ],
        options={'xscale': 'log', 'yscale': 'log'}
    )

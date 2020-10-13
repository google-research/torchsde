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

from tests.problems import Ex1
from torchsde import BrownianInterval
from torchsde.settings import LEVY_AREA_APPROXIMATIONS, SDE_TYPES
from . import inspection
from . import utils


def main():
    small_batch_size, large_batch_size, d = 16, 16384, 3
    t0, t1, steps, dt = 0., 2., 10, 1e-1
    ts = torch.linspace(t0, t1, steps=steps, device=device)
    dts = tuple(2 ** -i for i in range(1, 7))  # For checking strong order.
    sde = Ex1(d=d, sde_type=SDE_TYPES.stratonovich).to(device)
    methods = ('heun', 'euler_heun', 'midpoint', 'milstein', 'milstein')
    options = (None, None, None, None, dict(grad_free=True))
    labels = ('heun', 'euler-heun', 'midpoint', 'milstein', 'grad-free milstein')
    img_dir = os.path.join(os.path.dirname(__file__), 'plots', 'stratonovich_diagonal')

    y0 = torch.full((small_batch_size, d), fill_value=0.1, device=device)
    bm = BrownianInterval(
        t0=t0, t1=t1, shape=(small_batch_size, d), dtype=y0.dtype, device=device,
        levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time
    )
    inspection.inspect_samples(y0, ts, dt, sde, bm, img_dir, methods, options=options, labels=labels)

    y0 = torch.full((large_batch_size, d), fill_value=0.1, device=device)
    bm = BrownianInterval(
        t0=t0, t1=t1, shape=(large_batch_size, d), dtype=y0.dtype, device=device,
        levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time
    )
    inspection.inspect_orders(y0, t0, t1, dts, sde, bm, img_dir, methods, options=options, labels=labels)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    utils.manual_seed()

    main()

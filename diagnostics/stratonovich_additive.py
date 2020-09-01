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

from tests.basic_sde import AdditiveSDE
from torchsde import BrownianInterval
from torchsde.settings import LEVY_AREA_APPROXIMATIONS, SDE_TYPES
from . import inspect
from . import utils


def main():
    small_batch_size, large_batch_size, d, m = 16, 8192, 3, 5
    t0, t1, steps, dt = 0., 2., 10, 1e-1
    ts = torch.linspace(t0, t1, steps=steps, device=device)
    dts = tuple(2 ** -i for i in range(1, 8))  # For checking strong order.
    sde = AdditiveSDE(d=d, m=m, sde_type=SDE_TYPES.stratonovich).to(device)
    # Don't test Milstein methods, since there's no advantage to use extra resource to compute 0s.
    methods = ('heun', 'euler_heun', 'midpoint')
    img_dir = os.path.join('.', 'diagnostics', 'plots', 'stratonovich_additive')

    y0 = torch.full((small_batch_size, d), fill_value=0.1, device=device)
    bm = BrownianInterval(
        t0=t0, t1=t1, shape=(small_batch_size, m), dtype=y0.dtype, device=device,
        levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time, pool_size=16
    )
    inspect.inspect_samples(y0, ts, dt, sde, bm, img_dir, methods)

    y0 = torch.full((large_batch_size, d), fill_value=0.1, device=device)
    bm = BrownianInterval(
        t0=t0, t1=t1, shape=(large_batch_size, m), dtype=y0.dtype, device=device,
        levy_area_approximation=LEVY_AREA_APPROXIMATIONS.space_time, pool_size=16
    )
    inspect.inspect_strong_order(y0, t0, t1, dts, sde, bm, img_dir, methods)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    utils.manual_seed()

    main()

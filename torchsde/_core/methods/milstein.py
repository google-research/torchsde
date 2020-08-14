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

from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS

from .. import base_solver


class Milstein(base_solver.BaseSDESolver):
    strong_order = 1.0
    weak_order = 1.0
    sde_type = SDE_TYPES.ito
    noise_types = (NOISE_TYPES.additive, NOISE_TYPES.diagonal, NOISE_TYPES.scalar)
    levy_area = LEVY_AREA_APPROXIMATIONS.none

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        I_k = self.bm(t0, t0 + dt)
        v = [delta_bm_ ** 2. - dt for delta_bm_ in I_k]

        f_eval = self.sde.f(t0, y0)
        g_prod_eval = self.sde.g_prod(t0, y0, I_k)
        if self.sde.noise_type == NOISE_TYPES.additive:
            gdg_prod_eval = [0] * len(g_prod_eval)
        else:
            gdg_prod_eval = self.sde.gdg_prod(t0, y0, v)
        y1 = [
            y0_i + f_eval_i * dt + g_prod_eval_i + .5 * gdg_prod_eval_i
            for y0_i, f_eval_i, g_prod_eval_i, gdg_prod_eval_i in zip(y0, f_eval, g_prod_eval, gdg_prod_eval)
        ]
        t1 = t0 + dt
        return t1, y1

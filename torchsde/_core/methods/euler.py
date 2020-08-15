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


class Euler(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.ito
    noise_types = (NOISE_TYPES.additive, NOISE_TYPES.diagonal, NOISE_TYPES.general, NOISE_TYPES.scalar)
    levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none

    def __init__(self, sde, **kwargs):
        if sde.noise_type == NOISE_TYPES.additive:
            self.strong_order = 1.0
        else:
            self.strong_order = 0.5
        super(Euler, self).__init__(sde=sde, **kwargs)

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        t1 = t0 + dt

        I_k = self.bm(t0, t1)

        f_eval = self.sde.f(t0, y0)
        g_prod_eval = self.sde.g_prod(t0, y0, I_k)

        y1 = [
            y0_ + f_eval_ * dt + g_prod_eval_
            for y0_, f_eval_, g_prod_eval_ in zip(y0, f_eval, g_prod_eval)
        ]

        return t1, y1

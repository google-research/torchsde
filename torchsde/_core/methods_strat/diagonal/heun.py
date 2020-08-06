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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchsde._core import base_solver


class HeunDiagonal(base_solver.GenericSDESolver):

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        I_k = [(bm_next - bm_cur).to(y0[0]) for bm_next, bm_cur in zip(self.bm(t0 + dt), self.bm(t0))]

        f_eval = self.sde.f(t0, y0)
        g_eval = self.sde.g(t0, y0)
        g_prod_eval = self.sde.g_prod(g_eval, I_k)
        y1 = [
            y0_i + f_eval_i * dt + g_prod_eval_i
            for y0_i, f_eval_i, g_prod_eval_i in zip(y0, f_eval, g_prod_eval)
        ]
        
        t1 = t0 + dt

        f_eval_next_step_sum = self.sde.f(t1, y1) + f_eval
        g_eval_next_step_sum = self.sde.g(t1, y1) + g_eval
        g_prod_eval_next_step = self.sde.g_prod(g_eval_next_step_sum, I_k)
        y2 = [
            y0_i + 0.5 * (f_eval_i * dt + g_prod_eval_i)
            for y0_i, f_eval_i, g_prod_eval_i in zip(y1, f_eval_next_step_sum, g_prod_eval_next_step)
        ]

        return t1, y2

    @property
    def strong_order(self):
        return 0.5

    @property
    def weak_order(self):
        return 1.0

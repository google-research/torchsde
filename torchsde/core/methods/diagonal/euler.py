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

import torch

from torchsde.core import base_solver


class EulerDiagonal(base_solver.GenericSDESolver):

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        with torch.no_grad():
            I_k = tuple((bm_next - bm_cur).to(y0[0]) for bm_next, bm_cur in zip(self.bm(t0 + dt), self.bm(t0)))

        f_eval = self.sde.f(t0, y0)
        g_prod_eval = self.sde.g_prod(t0, y0, I_k)
        y1 = tuple(
            y0_i + f_eval_i * dt + g_prod_eval_i
            for y0_i, f_eval_i, g_prod_eval_i in zip(y0, f_eval, g_prod_eval)
        )
        t1 = t0 + dt
        return t1, y1

    @property
    def strong_order(self):
        return 0.5

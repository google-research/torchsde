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

import math

import torch

from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS, METHOD_OPTIONS as opt

from .. import base_solver


class BaseMilstein(base_solver.BaseSDESolver):
    strong_order = 1.0
    weak_order = 1.0
    noise_types = (NOISE_TYPES.additive, NOISE_TYPES.diagonal, NOISE_TYPES.scalar)
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def v_term(self, I_k, dt):
        pass

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        t1 = t0 + dt

        I_k = self.bm(t0, t1)
        v = self.v_term(I_k, dt)

        f_eval = self.sde.f(t0, y0)
        g_eval = self.sde.g(t0, y0)
        g_prod_eval = self.sde.prod(g_eval, I_k)

        if opt.grad_free in self.options and self.options[opt.grad_free]:
            g_prod_eval_v = self.sde.prod(g_eval, v)
            sqrt_dt = torch.sqrt(dt) if isinstance(dt, torch.Tensor) else math.sqrt(dt)
            y0_prime = [
                y0_ + dt * f_eval_ + g_eval_ * sqrt_dt
                for y0_, f_eval_, g_eval_ in zip(y0, f_eval, g_eval)
            ]            
            g_prod_eval_prime = self.sde.g_prod(t1, y0_prime, v)
            gdg_prod_eval = [
                (g_prod_eval_prime_ - g_prod_eval_v_) / sqrt_dt
                for g_prod_eval_prime_, g_prod_eval_v_ in zip(g_prod_eval_prime, g_prod_eval_v)
            ]
        else:
            gdg_prod_eval = self.sde.gdg_prod(t0, y0, v)
        
        y1 = [
            y0_i + f_eval_i * dt + g_prod_eval_i + .5 * gdg_prod_eval_i
            for y0_i, f_eval_i, g_prod_eval_i, gdg_prod_eval_i in zip(y0, f_eval, g_prod_eval, gdg_prod_eval)
        ]
        return t1, y1


class MilsteinIto(BaseMilstein):
    sde_type = SDE_TYPES.ito
    
    def v_term(self, I_k, dt):
        return [delta_bm_ ** 2. - dt for delta_bm_ in I_k]


class MilsteinStratonovich(BaseMilstein):
    sde_type = SDE_TYPES.stratonovich

    def v_term(self, I_k, dt):
        return [delta_bm_ ** 2. for delta_bm_ in I_k]

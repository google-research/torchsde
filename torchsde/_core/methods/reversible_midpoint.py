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

from .. import adjoint_sde
from .. import base_solver
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS, METHODS


class ReversibleMidpoint(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        self.strong_order = 0.5 if sde.noise_type == NOISE_TYPES.general else 1.0
        super(ReversibleMidpoint, self).__init__(sde=sde, **kwargs)

    def _init_extra_solver_state(self, t0, y0):
        return self.sde.f_and_g(t0, y0)

    def step(self, t0, t1, y0, extra0):
        f0, g0 = extra0
        dt = t1 - t0
        dW = self.bm(t0, t1)

        half_dt = 0.5 * dt
        t_prime = t0 + half_dt
        y_prime = y0 + half_dt * f0 + 0.5 * self.sde.prod(g0, dW)

        f_prime, g_prime = self.sde.f_and_g(t_prime, y_prime)

        y1 = y0 + dt * f_prime + self.sde.prod(g_prime, dW)
        f1 = 2 * f_prime - f0
        g1 = 2 * g_prime - g0

        return y1, (f1, g1)


class AdjointReversibleMidpoint(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        if not isinstance(sde, adjoint_sde.AdjointSDE):
            raise ValueError(f"{METHODS.adjoint_reversible_midpoint} can only be used for adjoint_method.")
        self.strong_order = 0.5 if sde.noise_type == NOISE_TYPES.general else 1.0
        super(AdjointReversibleMidpoint, self).__init__(sde=sde, **kwargs)
        self.forward_sde = sde.base_sde

    def _init_extra_solver_state(self, t0, y0):
        # We expect to always be given the extra state from the forward pass.
        raise RuntimeError("Please report a bug to torchsde.")

    def step(self, t0, t1, y0, extra0):
        f0, forward_g0 = extra0
        dt = t1 - t0
        dW = self.bm(t0, t1)
        forward_g_prod0 = self.forward_sde.prod(forward_g0, dW)

        forward_y0, adj_y0, requires_grad = self.sde._get_state(t0, y0)

        with torch.enable_grad():
            vjp_y_and_params = misc.vjp(
                # Have to re-evaluate the diffusion of the forward SDE, so that we can calculate its gradient.
                outputs=self.forward_sde.g_prod(t0, y0, dW),
                inputs=[forward_y0] + self.sde._params,
                grad_outputs=adj_y0,
                allow_unused=True,
                retain_graph=True,
                create_graph=requires_grad
            )
            g_prod0 = misc.flatten((-forward_g_prod0, *vjp_y_and_params))

        half_dt = 0.5 * dt
        t_prime = t0 + half_dt
        y_prime = y0 + half_dt * f0 + 0.5 * g_prod0

        forward_y_prime, adj_y_prime, _ = self.sde._get_state(t_prime, y_prime)
        with torch.enable_grad():
            forward_f_prime, forward_g_prime = self.forward_sde.f_and_g(t_prime, forward_y_prime)
            forward_g_prod_prime = self.forward_sde.prod(forward_g_prime, dW)
            f_prime = self.sde._f_uncorrected(forward_f_prime, forward_y_prime, adj_y_prime, requires_grad)
            g_prod_prime = self.sde._g_prod(forward_g_prod_prime, forward_y_prime, adj_y_prime, requires_grad)

        y1 = y0 + dt * f_prime + g_prod_prime
        f1 = 2 * f_prime - f0
        forward_g1 = 2 * forward_g_prime - forward_g0

        return y1, (f1, forward_g1)

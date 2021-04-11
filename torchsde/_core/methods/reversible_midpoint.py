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

import torch

from .. import adjoint_sde
from .. import base_solver
from .. import misc
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS, METHODS


class ReversibleMidpoint(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        self.strong_order = 0.5 if sde.noise_type == NOISE_TYPES.general else 1.0
        super(ReversibleMidpoint, self).__init__(sde=sde, **kwargs)

    def init_extra_solver_state(self, t0, y0):
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

        if self.forward_sde.noise_type == NOISE_TYPES.diagonal:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1 * tensor2
        else:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1.unsqueeze(-1) * tensor2.unsqueeze(-2)

    def init_extra_solver_state(self, t0, y0):
        # We expect to always be given the extra state from the forward pass.
        raise RuntimeError("Please report a bug to torchsde.")

    # So deriving the algebra here is not straightforward.
    #
    # This implementation was derived by starting with an implementation that (a) calculated the backward step for the
    # original state, (b) made a local forward, (c) then made a local backward. This involved two evaluations each for
    # the vector fields f and g: one in (a) and one in (b). Then I (Patrick) simply made program transformations,
    # manually calculating how each one affected the gradients, until the two vector field evaluations could be merged
    # into one (and the autograd only occuring around the user-supplied vector fields). Thus improving computational
    # efficiency quite meaningfully.
    #
    # I do not claim that this is particularly readable, however.
    def step(self, t0, t1, y0, extra0):
        forward_f0, forward_g0 = extra0
        dt = t1 - t0
        dW = self.bm(t0, t1)
        half_dt = 0.5 * dt
        half_dW = 0.5 * dW
        forward_y0, adj_y0, (adj_f0, adj_g0), requires_grad = self.sde.get_state(t0, y0)
        twice_adj_f0 = 2 * adj_f0
        twice_adj_g0 = 2 * adj_g0

        t_prime = t0 + half_dt
        forward_y_prime = forward_y0 - half_dt * forward_f0 - self.forward_sde.prod(forward_g0, half_dW)

        with torch.enable_grad():
            if not forward_y_prime.requires_grad:
                forward_y_prime = forward_y_prime.detach().requires_grad_()
            forward_f_prime, forward_g_prime = self.forward_sde.f_and_g(-t_prime, forward_y_prime)
            vjp_y, *vjp_params = misc.vjp(outputs=(forward_f_prime, forward_g_prime),
                                          inputs=[forward_y_prime] + self.sde.params,
                                          grad_outputs=[twice_adj_f0 + dt * adj_y0,
                                                        twice_adj_g0 + self._adjoint_of_prod(adj_y0, dW)],
                                          allow_unused=True,
                                          create_graph=requires_grad)
            if not requires_grad:
                forward_f_prime = forward_f_prime.detach()
                forward_g_prime = forward_g_prime.detach()

        forward_dy1 = dt * forward_f_prime + self.forward_sde.prod(forward_g_prime, dW)
        vjp_f = half_dt * vjp_y - twice_adj_f0
        vjp_g = self._adjoint_of_prod(vjp_y, half_dW) - twice_adj_g0
        dy1 = misc.flatten([-forward_dy1, vjp_y, vjp_f, vjp_g] + vjp_params)

        y1 = y0 + dy1
        forward_f1 = 2 * forward_f_prime - forward_f0
        forward_g1 = 2 * forward_g_prime - forward_g0

        return y1, (forward_f1, forward_g1)

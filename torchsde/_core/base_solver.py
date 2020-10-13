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

import abc
import warnings

import torch

from . import adaptive_stepping
from . import better_abc
from . import interp
from . import misc
from .base_sde import BaseSDE
from .._brownian import BaseBrownian
from ..settings import NOISE_TYPES
from ..types import Scalar, Tensor, Dict


class BaseSDESolver(metaclass=better_abc.ABCMeta):
    """API for solvers with possibly adaptive time stepping."""

    strong_order = better_abc.abstract_attribute()
    weak_order = better_abc.abstract_attribute()
    sde_type = better_abc.abstract_attribute()
    noise_types = better_abc.abstract_attribute()
    levy_area_approximations = better_abc.abstract_attribute()

    def __init__(self,
                 sde: BaseSDE,
                 bm: BaseBrownian,
                 y0: Tensor,
                 dt: Scalar,
                 adaptive: bool,
                 rtol: Scalar,
                 atol: Scalar,
                 dt_min: Scalar,
                 options: Dict,
                 **kwargs):
        super(BaseSDESolver, self).__init__(**kwargs)
        assert sde.sde_type == self.sde_type, f"SDE is of type {sde.sde_type} but solver is for type {self.sde_type}"
        assert sde.noise_type in self.noise_types, (
            f"SDE has noise type {sde.noise_type} but solver only supports noise types {self.noise_types}"
        )
        assert bm.levy_area_approximation in self.levy_area_approximations, (
            f"SDE solver requires one of {self.levy_area_approximations} set as the `levy_area_approximation` on the "
            f"Brownian motion."
        )
        if sde.noise_type == NOISE_TYPES.scalar and torch.Size(bm.shape[1:]).numel() != 1:  # noqa
            raise ValueError("The Brownian motion for scalar SDEs must of dimension 1.")

        self.sde = sde
        self.bm = bm
        self.y0 = y0
        self.dt = dt
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.options = options

    def __repr__(self):
        return f"{self.__class__.__name__} of strong order: {self.strong_order}, and weak order: {self.weak_order}"

    @abc.abstractmethod
    def step(self, t0: Scalar, t1: Scalar, y0: Tensor) -> Tensor:
        """Propose a step with step size from time t to time next_t, with
         current state y.

        Args:
            t0: float or Tensor of size (,).
            t1: float or Tensor of size (,).
            y0: Tensor of size (batch_size, d).

        Returns:
            y1, where y1 is a Tensor of size (batch_size, d).
        """
        raise NotImplementedError

    def integrate(self, ts: Tensor) -> Tensor:
        """Integrate along trajectory.

        Args:
            ts: Tensor of size (T,).

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
        """
        assert misc.is_strictly_increasing(ts), "Evaluation times `ts` must be strictly increasing."
        y0, dt, adaptive, rtol, atol, dt_min = self.y0, self.dt, self.adaptive, self.rtol, self.atol, self.dt_min

        step_size = dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0

        ys = [y0]
        prev_error_ratio = None

        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                if adaptive:
                    # Take 1 full step.
                    next_y_full = self.step(curr_t, next_t, curr_y)
                    # Take 2 half steps.
                    midpoint_t = 0.5 * (curr_t + next_t)
                    midpoint_y = self.step(curr_t, midpoint_t, curr_y)
                    next_y = self.step(midpoint_t, next_t, midpoint_y)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, rtol, atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_step_size(
                            error_estimate=error_estimate,
                            prev_step_size=step_size,
                            prev_error_ratio=prev_error_ratio
                        )

                    if step_size < dt_min:
                        warnings.warn("Hitting minimum allowed step size in adaptive time-stepping.")
                        step_size = dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= dt_min:
                        prev_t, prev_y = curr_t, curr_y
                        curr_t, curr_y = next_t, next_y
                else:
                    prev_t, prev_y = curr_t, curr_y
                    curr_t, curr_y = next_t, self.step(curr_t, next_t, curr_y)
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0)

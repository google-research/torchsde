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

from ..settings import NOISE_TYPES

from . import adaptive_stepping
from . import better_abc
from . import interp
from . import misc


class BaseSDESolver(metaclass=better_abc.ABCMeta):
    """API for solvers with possibly adaptive time stepping."""

    strong_order = better_abc.abstract_attribute()
    weak_order = better_abc.abstract_attribute()
    sde_type = better_abc.abstract_attribute()
    noise_types = better_abc.abstract_attribute()
    levy_area_approximations = better_abc.abstract_attribute()

    def __init__(self, sde, bm, y0, dt, adaptive, rtol, atol, dt_min, options, **kwargs):
        super(BaseSDESolver, self).__init__(**kwargs)
        assert misc.is_seq_not_nested(y0), 'Initial value for integration should be a tuple of tensors.'
        assert sde.sde_type == self.sde_type, f"SDE is of type {sde.sde_type} but solver is for type {self.sde_type}"
        assert sde.noise_type in self.noise_types, (
            f"SDE has noise type {sde.noise_type} but solver only supports noise types {self.noise_types}"
        )
        assert bm.levy_area_approximation in self.levy_area_approximations, (
            f"SDE solver requires one of {self.levy_area_approximations} set as the `levy_area_approximation` on the "
            f"Brownian motion."
        )
        if sde.noise_type == NOISE_TYPES.scalar and torch.Size(bm.shape[1:]).numel() != 1:
            raise ValueError('The Brownian motion for scalar SDEs must of dimension 1.')

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
        return f'{self.__class__.__name__} of strong order: {self.strong_order}'

    @abc.abstractmethod
    def step(self, t, y, dt):
        """Propose a step with step size dt, starting at time t and state y.

        Args:
            t: float or torch.Tensor of size (,).
            y: torch.Tensor of size (batch_size, d).
            dt: float or torch.Tensor of size (,).

        Returns:
            (t1, y1), where t1 is a float or torch.Tensor of size (,)
            and y1 is a torch.Tensor of size (batch_size, d).
        """
        raise NotImplementedError

    def step_logqp(self, t, next_t, y, logqp0):
        dt = next_t - t
        y1 = self.step_(t, next_t, y)

        if self.sde.noise_type in ("diagonal", "scalar"):
            f_eval = self.sde.f(t, y)
            g_eval = self.sde.g(t, y)
            h_eval = self.sde.h(t, y)
            u_eval = misc.seq_sub_div(f_eval, h_eval, g_eval)
            logqp1 = [
                logqp0_i + .5 * torch.sum(u_eval_i ** 2., dim=1) * dt
                for logqp0_i, u_eval_i in zip(logqp0, u_eval)
            ]
        else:
            f_eval = self.sde.f(t, y)
            g_eval = self.sde.g(t, y)
            h_eval = self.sde.h(t, y)

            g_inv_eval = [torch.pinverse(g_eval_) for g_eval_ in g_eval]
            u_eval = misc.seq_sub(f_eval, h_eval)
            u_eval = misc.seq_batch_mvp(ms=g_inv_eval, vs=u_eval)
            logqp1 = [
                logqp0_i + .5 * torch.sum(u_eval_i ** 2., dim=1) * dt
                for logqp0_i, u_eval_i in zip(logqp0, u_eval)
            ]
        return y1, logqp1

    # TODO: adjust solvers and remove this
    def step_(self, t, next_t, y):
        dt = next_t - t
        _, next_y = self.step(t, y, dt)
        return next_y

    # TODO: unify integrate and integrate_logqp? My IDE spits out so many warnings about duplicate code.
    def integrate(self, ts):
        """Integrate along trajectory.

        Returns:
            A single state tensor of size (T, batch_size, d) (or tuple).
        """
        assert misc.is_increasing(ts), 'Evaluation timestamps should be strictly increasing.'
        y0, dt, adaptive, rtol, atol, dt_min = (self.y0, self.dt, self.adaptive, self.rtol, self.atol, self.dt_min)

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
                    next_y_full = self.step_(curr_t, next_t, curr_y)
                    # Take 2 half steps.
                    midpoint_t = 0.5 * (curr_t + next_t)
                    midpoint_y = self.step_(curr_t, midpoint_t, curr_y)
                    next_y = self.step_(midpoint_t, next_t, midpoint_y)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, rtol, atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_step_size(
                            error_estimate=error_estimate,
                            prev_step_size=step_size,
                            prev_error_ratio=prev_error_ratio
                        )

                    if step_size < dt_min:
                        warnings.warn('Hitting minimum allowed step size in adaptive time-stepping.')
                        step_size = dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= dt_min:
                        prev_t, prev_y = curr_t, curr_y
                        curr_t, curr_y = next_t, next_y
                else:
                    prev_t, prev_y = curr_t, curr_y
                    curr_t, curr_y = next_t, self.step_(curr_t, next_t, curr_y)
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        ans = tuple(torch.stack([ys[j][i] for j in range(len(ts))], dim=0) for i in range(len(y0)))
        return ans

    def integrate_logqp(self, ts):
        """Integrate along trajectory; also return the log-ratio.

        Returns:
            A single state tensor of size (T, batch_size, d) (or tuple), and a single log-ratio tensor of
            size (T - 1, batch_size) (or tuple).
        """
        assert misc.is_increasing(ts), 'Evaluation timestamps should be strictly increasing.'
        y0, dt, adaptive, rtol, atol, dt_min = (self.y0, self.dt, self.adaptive, self.rtol, self.atol, self.dt_min)

        step_size = dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0

        ys = [y0]
        prev_error_ratio = None
        logqp = [[] for _ in y0]

        for out_t in ts[1:]:
            curr_logqp = [0. for _ in y0]
            prev_logqp = curr_logqp
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                if adaptive:
                    # Take 1 full step.
                    next_y_full = self.step_(curr_t, next_t, curr_y)
                    # Take 2 half steps.
                    midpoint_t = 0.5 * (curr_t + next_t)
                    midpoint_y, midpoint_logqp = self.step_logqp(curr_t, midpoint_t, curr_y, curr_logqp)
                    next_y, next_logqp = self.step_logqp(midpoint_t, next_t, midpoint_y, midpoint_logqp)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, rtol, atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_step_size(
                            error_estimate=error_estimate,
                            prev_step_size=step_size,
                            prev_error_ratio=prev_error_ratio
                        )

                    if step_size < dt_min:
                        warnings.warn('Hitting minimum allowed step size in adaptive time-stepping.')
                        step_size = dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= dt_min:
                        prev_t, prev_y, prev_logqp = curr_t, curr_y, curr_logqp
                        curr_t, curr_y, curr_logqp = next_t, next_y, next_logqp
                else:
                    prev_t, prev_y, prev_logqp = curr_t, curr_y, curr_logqp
                    curr_y, curr_logqp = self.step_logqp(curr_t, next_t, curr_y, curr_logqp)
                    curr_t = next_t
            ret_y, ret_logqp = interp.linear_interp_logqp(t0=prev_t, y0=prev_y, logqp0=prev_logqp, t1=curr_t,
                                                          y1=curr_y, logqp1=curr_logqp, t=out_t)
            ys.append(ret_y)
            [logqp_i.append(ret_logqp_i) for logqp_i, ret_logqp_i in zip(logqp, ret_logqp)]

        ans = [torch.stack([ys[j][i] for j in range(len(ts))], dim=0) for i in range(len(y0))]
        logqp = [torch.stack(logqp_i, dim=0) for logqp_i in logqp]
        return (*ans, *logqp)

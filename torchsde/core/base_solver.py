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

import abc
import warnings

import torch

from torchsde.core import adaptive_stepping
from torchsde.core import misc


class SDESolver(abc.ABC):
    """Abstract class specifying the methods that must be implemented for any solver."""

    @abc.abstractmethod
    def integrate(self, ts):
        pass


class GenericSDESolver(SDESolver):
    """API for solvers with possibly adaptive time stepping."""

    def __init__(self, sde, bm, y0, dt, adaptive, rtol, atol, dt_min, options):
        super(GenericSDESolver, self).__init__()
        assert misc.is_seq_not_nested(y0), 'Initial value for integration should be a tuple of tensors.'
        self.sde = sde
        self.bm = bm
        self.y0 = y0
        self.dt = dt
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.options = options

    @property
    @abc.abstractmethod
    def strong_order(self):
        pass

    @property
    def weak_order(self):
        # TODO: Add weak orders for existing solvers.
        return None

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
        pass

    def step_logqp(self, t, y, dt, logqp0):
        t1, y1 = self.step(t, y, dt)

        if self.sde.noise_type in ("diagonal", "scalar"):
            f_eval = self.sde.f(t, y)
            g_eval = self.sde.g(t, y)
            h_eval = self.sde.h(t, y)
            u_eval = misc.seq_sub_div(f_eval, h_eval, g_eval)
            logqp1 = tuple(
                logqp0_i + .5 * torch.sum(u_eval_i ** 2., dim=1) * dt
                for logqp0_i, u_eval_i in zip(logqp0, u_eval)
            )
        else:
            f_eval = self.sde.f(t, y)
            g_eval = self.sde.g(t, y)
            h_eval = self.sde.h(t, y)

            ginv_eval = tuple(torch.pinverse(g_eval_) for g_eval_ in g_eval)
            u_eval = misc.seq_sub(f_eval, h_eval)
            u_eval = misc.seq_batch_mvp(ms=ginv_eval, vs=u_eval)
            logqp1 = tuple(
                logqp0_i + .5 * torch.sum(u_eval_i ** 2., dim=1) * dt
                for logqp0_i, u_eval_i in zip(logqp0, u_eval)
            )
        return t1, y1, logqp1

    def integrate(self, ts):
        """Integrate along trajectory.

        Returns:
            A single state tensor of size (T, batch_size, d) (or tuple).
        """
        assert misc.is_increasing(ts), 'Evaluation timestamps should be strictly increasing.'
        y0, dt, adaptive, rtol, atol, dt_min = (self.y0, self.dt, self.adaptive, self.rtol, self.atol, self.dt_min)

        step_size = dt
        curr_t = ts[0]
        curr_y = y0
        ys = [y0]
        prev_error_ratio = None
        prev_t, prev_y = curr_t, curr_y

        for next_t in ts[1:]:
            while curr_t < next_t:
                if adaptive:
                    delta_t = step_size
                    # Take 1 full step.
                    t1f, y1f = self.step(curr_t, curr_y, delta_t)
                    # Take 2 half steps.
                    t05, y05 = self.step(curr_t, curr_y, delta_t / 2)
                    t1h, y1h = self.step(t05, y05, delta_t / 2)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(y1f, y1h, rtol, atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_stepsize(
                            error_estimate=error_estimate, prev_stepsize=step_size, prev_error_ratio=prev_error_ratio)

                    if step_size < dt_min:
                        warnings.warn('Hitting minimum allowed step size in adaptive time-stepping.')
                        step_size = dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= dt_min:
                        prev_t, prev_y = curr_t, curr_y
                        curr_t, curr_y = t1h, y1h
                    del t1f, y1f
                else:
                    delta_t = step_size
                    prev_t, prev_y = curr_t, curr_y
                    curr_t, curr_y = self.step(curr_t, curr_y, delta_t)

            # Pull back when overshoot.
            curr_t, curr_y = self.step(prev_t, prev_y, next_t - prev_t)
            ys.append(curr_y)

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
        curr_t = ts[0]
        curr_y = y0
        ys = [y0]
        logqp = [[] for _ in y0]
        prev_error_ratio = None
        prev_t, prev_y = curr_t, curr_y

        for next_t in ts[1:]:
            curr_logqp = tuple(0. for _ in y0)
            prev_logqp = curr_logqp
            while curr_t < next_t:
                if adaptive:
                    delta_t = step_size
                    # Take 1 full step.
                    t1f, y1f, logqp1f = self.step_logqp(curr_t, curr_y, delta_t, logqp0=curr_logqp)
                    # Take 2 half steps.
                    t05, y05, logqp05 = self.step_logqp(curr_t, curr_y, delta_t / 2, logqp0=curr_logqp)
                    t1h, y1h, logqp1h = self.step_logqp(t05, y05, delta_t / 2, logqp0=logqp05)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(y1f, y1h, rtol, atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_stepsize(
                            error_estimate=error_estimate, prev_stepsize=step_size, prev_error_ratio=prev_error_ratio)

                    if step_size < dt_min:
                        warnings.warn('Hitting minimum allowed step size in adaptive time-stepping.')
                        step_size = dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= dt_min:
                        prev_t, prev_y, prev_logqp = curr_t, curr_y, curr_logqp
                        curr_t, curr_y, curr_logqp = t1h, y1h, logqp1h
                    del t1f, y1f, logqp1f
                else:
                    delta_t = step_size
                    prev_t, prev_y, prev_logqp = curr_t, curr_y, curr_logqp
                    curr_t, curr_y, curr_logqp = self.step_logqp(curr_t, curr_y, delta_t, logqp0=curr_logqp)

            # Pull back when overshoot.
            curr_t, curr_y, curr_logqp = self.step_logqp(prev_t, prev_y, next_t - prev_t, logqp0=prev_logqp)
            ys.append(curr_y)
            for logqp_i, curr_logqp_i in zip(logqp, curr_logqp):
                logqp_i.append(curr_logqp_i)

        ans = tuple(torch.stack([ys[j][i] for j in range(len(ts))], dim=0) for i in range(len(y0)))
        logqp = tuple(torch.stack(logqp_i, dim=0) for logqp_i in logqp)
        return (*ans, *logqp)

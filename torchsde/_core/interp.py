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


def linear_interp(t0, y0, t1, y1, t):
    assert t0 <= t <= t1, f'Incorrect time order for linear interpolation: t0={t0}, t={t}, t1={t1}.'
    y = [(t1 - t) / (t1 - t0) * y0_ + (t - t0) / (t1 - t0) * y1_ for y0_, y1_ in zip(y0, y1)]
    return y


def linear_interp_logqp(t0, y0, logqp0, t1, y1, logqp1, t):
    assert t0 <= t <= t1, f'Incorrect time order for linear interpolation: t0={t0}, t={t}, t1={t1}.'
    y = [(t1 - t) / (t1 - t0) * y0_ + (t - t0) / (t1 - t0) * y1_ for y0_, y1_ in zip(y0, y1)]
    logqp = [(t1 - t) / (t1 - t0) * l0 + (t - t0) / (t1 - t0) * l1 for l0, l1 in zip(logqp0, logqp1)]
    return y, logqp

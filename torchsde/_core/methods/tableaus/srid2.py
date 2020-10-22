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

# From "RUNGE-KUTTA METHODS FOR THE STRONG APPROXIMATION OF SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS".
# For diagonal noise structure.
# (ODE order, SDE strong order) = (3.0, 1.5).

STAGES = 4

C0 = (0, 1, 1 / 2, 0)
C1 = (0, 1 / 4, 1, 1 / 4)

A0 = (
    (),
    (1,),
    (1 / 4, 1 / 4),
    (0, 0, 0)
)
A1 = (
    (),
    (1 / 4,),
    (1, 0),
    (0, 0, 1 / 4)
)

B0 = (
    (),
    (0,),
    (1, 1 / 2),
    (0, 0, 0),
)
B1 = (
    (),
    (-1 / 2,),
    (1, 0),
    (2, -1, 1 / 2)
)

alpha = (1 / 6, 1 / 6, 2 / 3, 0)
beta1 = (-1, 4 / 3, 2 / 3, 0)
beta2 = (1, -4 / 3, 1 / 3, 0)
beta3 = (2, -4 / 3, -2 / 3, 0)
beta4 = (-2, 5 / 3, -2 / 3, 1)

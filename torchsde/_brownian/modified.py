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

from . import base_brownian


class _ModifiedBrownian(base_brownian.BaseBrownian):
    def __init__(self, base_brownian):
        super(_ModifiedBrownian, self).__init__()
        self.base_brownian = base_brownian

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.base_brownian)})"

    def to(self, *args, **kwargs):
        self.base_brownian.to(*args, **kwargs)

    @property
    def dtype(self):
        return self.base_brownian.dtype

    @property
    def device(self):
        return self.base_brownian.device

    @property
    def shape(self):
        return self.base_brownian.shape

    @property
    def levy_area_approximation(self):
        return self.base_brownian.levy_area_approximation


# TODO: these checks are very inelegant. Is there a better interface to Brownian motion?
class ReverseBrownian(_ModifiedBrownian):
    def __call__(self, ta, tb, return_U=False, return_A=False):
        # TODO: double-check if U and A need reversing
        if return_U:
            if return_A:
                W, U, A = self.base_brownian(-tb, -ta, return_U=return_U, return_A=return_A)
                return tuple(-W_ for W_ in W), U, A
            else:
                W, U = self.base_brownian(-tb, -ta, return_U=return_U, return_A=return_A)
                return tuple(-W_ for W_ in W), U
        else:
            if return_A:
                W, A = self.base_brownian(-tb, -ta, return_U=return_U, return_A=return_A)
                return tuple(-W_ for W_ in W), A
            else:
                W = self.base_brownian(-tb, -ta, return_U=return_U, return_A=return_A)
                return tuple(-W_ for W_ in W)


class TupleBrownian(_ModifiedBrownian):
    def __call__(self, ta, tb, return_U=False, return_A=False):
        if return_U:
            if return_A:
                W, U, A = self.base_brownian(ta, tb, return_U=return_U, return_A=return_A)
                return (W,), (U,), (A,)
            else:
                W, U = self.base_brownian(ta, tb, return_U=return_U, return_A=return_A)
                return (W,), (U,)
        else:
            if return_A:
                W, A = self.base_brownian(ta, tb, return_U=return_U, return_A=return_A)
                return (W,), (A,)
            else:
                W = self.base_brownian(ta, tb, return_U=return_U, return_A=return_A)
                return (W,)

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

from . import base_brownian
from .._core.settings import LEVY_AREA_APPROXIMATIONS


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


class ReverseBrownian(_ModifiedBrownian):
    def __call__(self, ta, tb):
        if self.levy_area_approximation == LEVY_AREA_APPROXIMATIONS.none:
            return tuple(-b for b in self.base_brownian(-tb, -ta))
        else:
            return tuple((-b, -h, a) for b, h, a in self.base_brownian(-tb, -ta))


class TupleBrownian(_ModifiedBrownian):
    def __call__(self, ta, tb):
        return (self.base_brownian(ta, tb),)


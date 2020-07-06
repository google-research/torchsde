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

import unittest

import numpy as np
import numpy.random as npr
import torch
from scipy.stats import norm, kstest

from torchsde import BrownianPath
from .torch_test import TorchTestCase

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

D = 3
BATCH_SIZE = 131072
REPS = 3
ALPHA = 0.001


class TestBrownianPath(TorchTestCase):

    def _setUp(self, device=None):
        t0, t1 = torch.tensor([0., 1.]).to(device)
        w0, w1 = torch.randn([2, BATCH_SIZE, D]).to(device)

        self.t = torch.rand([]).to(device)
        self.bm = BrownianPath(t0=t0, w0=w0)

    def test_basic_cpu(self):
        self._setUp(device=torch.device('cpu'))
        sample = self.bm(self.t)
        self.assertEqual(sample.size(), (BATCH_SIZE, D))

    def test_basic_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest(reason='CUDA not available.')

        self._setUp(device=torch.device('cuda'))
        sample = self.bm(self.t)
        self.assertEqual(sample.size(), (BATCH_SIZE, D))

    def test_determinism(self):
        self._setUp()
        vals = [self.bm(self.t) for _ in range(REPS)]
        for val in vals[1:]:
            self.tensorAssertAllClose(val, vals[0])

    def test_normality(self):
        """Kolmogorov-Smirnov test."""
        t0_, t1_ = 0.0, 1.0
        t0, t1 = torch.tensor([t0_, t1_])
        eps = 1e-2
        for _ in range(REPS):
            w0_, w1_ = 0.0, npr.randn() * np.sqrt(t1_)
            # Use the same endpoint for the batch, so samples from same dist.
            w0 = torch.tensor(w0_).repeat(BATCH_SIZE)
            w1 = torch.tensor(w1_).repeat(BATCH_SIZE)

            bm = BrownianPath(t0=t0, w0=w0)
            bm.insert(t=t1, w=w1)

            t_ = npr.uniform(low=t0_ + eps, high=t1_ - eps)
            samples = bm(t_)
            samples_ = samples.detach().numpy()

            mean_ = ((t1_ - t_) * w0_ + (t_ - t0_) * w1_) / (t1_ - t0_)
            std_ = np.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
            ref_dist = norm(loc=mean_, scale=std_)

            _, pval = kstest(samples_, ref_dist.cdf)
            self.assertGreaterEqual(pval, ALPHA)


if __name__ == '__main__':
    unittest.main()

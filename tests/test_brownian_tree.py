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

import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import unittest

import numpy as np
import numpy.random as npr
import torch
from scipy.stats import norm, kstest

from tests.torch_test import TorchTestCase
from torchsde.brownian_lib import BrownianTree

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

D = 3
SMALL_BATCH_SIZE = 16
LARGE_BATCH_SIZE = 16384
REPS = 3
ALPHA = 0.001


class TestBrownianTree(TorchTestCase):

    def _setUp(self, batch_size, device=None):
        t0, t1 = torch.tensor([0., 1.]).to(device)
        w0 = torch.zeros(batch_size, D).to(device=device)
        t = torch.rand([]).to(device)

        self.t = t
        self.bm = BrownianTree(t0=t0, t1=t1, w0=w0, entropy=0)

    def test_basic_cpu(self):
        self._setUp(batch_size=SMALL_BATCH_SIZE, device=torch.device('cpu'))
        sample = self.bm(self.t)
        self.assertEqual(sample.size(), (SMALL_BATCH_SIZE, D))

    def test_basic_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest(reason='CUDA not available.')

        self._setUp(batch_size=SMALL_BATCH_SIZE, device=torch.device('cuda'))
        sample = self.bm(self.t)
        self.assertEqual(sample.size(), (SMALL_BATCH_SIZE, D))

    def test_determinism(self):
        self._setUp(batch_size=SMALL_BATCH_SIZE)
        vals = [self.bm(self.t) for _ in range(REPS)]
        for val in vals[1:]:
            self.tensorAssertAllClose(val, vals[0])

    def test_normality(self):
        """Kolmogorov-Smirnov test."""
        t0_, t1_ = 0.0, 1.0
        t0, t1 = torch.tensor([t0_, t1_])
        eps = 1e-5
        for _ in range(REPS):
            w0_, w1_ = 0.0, npr.randn()
            # Use the same endpoint for the batch, so samples from same dist.
            w0 = torch.tensor(w0_).repeat(LARGE_BATCH_SIZE)
            w1 = torch.tensor(w1_).repeat(LARGE_BATCH_SIZE)
            bm = BrownianTree(t0=t0, t1=t1, w0=w0, w1=w1, pool_size=100, tol=1e-14)

            for _ in range(REPS):
                t_ = npr.uniform(low=t0_ + eps, high=t1_ - eps)
                samples = bm(t_)
                samples_ = samples.detach().numpy()

                mean_ = ((t1_ - t_) * w0_ + (t_ - t0_) * w1_) / (t1_ - t0_)
                std_ = np.sqrt((t1_ - t_) * (t_ - t0_) / (t1_ - t0_))
                ref_dist = norm(loc=mean_, scale=std_)

                _, pval = kstest(samples_, ref_dist.cdf)
                self.assertGreaterEqual(pval, ALPHA)

    def test_to_device(self):
        if not torch.cuda.is_available():
            self.skipTest(reason='CUDA not available.')

        self._setUp(batch_size=SMALL_BATCH_SIZE)
        curr, prev, post = _dict_to_sorted_list(*self.bm.get_cache())
        old = torch.cat(curr + prev + post, dim=0)

        self.bm.to(torch.device('cuda'))
        curr, prev, post = _dict_to_sorted_list(*self.bm.get_cache())
        new = torch.cat(curr + prev + post, dim=0)
        self.assertTrue(str(new.device).startswith('cuda'))
        self.tensorAssertAllClose(old, new.cpu())

    def test_to_float32(self):
        self._setUp(batch_size=SMALL_BATCH_SIZE)
        curr, prev, post = _dict_to_sorted_list(*self.bm.get_cache())
        old = torch.cat(curr + prev + post, dim=0)

        self.bm.to(torch.float32)
        curr, prev, post = _dict_to_sorted_list(*self.bm.get_cache())
        new = torch.cat(curr + prev + post, dim=0)
        self.assertTrue(new.dtype, torch.float32)
        self.tensorAssertAllClose(old, new.double())


def _dict_to_sorted_list(*dicts):
    lists = tuple([d[k] for k in sorted(d.keys())] for d in dicts)
    if len(lists) == 1:
        return lists[0]
    return lists


if __name__ == '__main__':
    unittest.main()

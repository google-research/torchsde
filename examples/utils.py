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

import argparse
import os

import torch


def makedirs(*dirs):
    for d in dirs:
        assert isinstance(d, str)
        if not os.path.exists(d):
            os.makedirs(d)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        iters = max(1, iters)
        self.val = maxval / iters
        self.maxval = maxval
        self.iters = iters

    def step(self):
        self.val = min(self.maxval, self.val + self.maxval / self.iters)

    def __call__(self):
        return self.val


class EMAMetric(object):
    def __init__(self, gamma=.99):
        super(EMAMetric, self).__init__()
        self.prev_metric = 0.
        self.gamma = gamma

    def step(self, x):
        with torch.no_grad():
            self.prev_metric = (1. - self.gamma) * self.prev_metric + self.gamma * x

    def val(self):
        return self.prev_metric

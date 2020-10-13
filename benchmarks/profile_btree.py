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

import logging
import os
import time

import matplotlib.pyplot as plt
import torch
import tqdm

from torchsde import BrownianTree


def run_torch(ks=(0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12)):
    w0 = torch.zeros(b, d)

    t_cons = []
    t_queries = []
    t_alls = []
    for k in tqdm.tqdm(ks):
        now = time.time()
        bm_vanilla = BrownianTree(t0=t0, t1=t1, w0=w0, cache_depth=k)
        t_con = time.time() - now
        t_cons.append(t_con)

        now = time.time()
        for t in ts:
            bm_vanilla(t).to(device)
        t_query = time.time() - now
        t_queries.append(t_query)

        t_all = t_con + t_query
        t_alls.append(t_all)
        logging.warning(f'k={k}, t_con={t_con:.4f}, t_query={t_query:.4f}, t_all={t_all:.4f}')

    img_path = os.path.join('.', 'diagnostics', 'plots', 'profile_btree.png')
    plt.figure()
    plt.plot(ks, t_cons, label='cons')
    plt.plot(ks, t_queries, label='queries')
    plt.plot(ks, t_alls, label='all')
    plt.title(f'b={b}, d={d}, repetitions={reps}, device={w0.device}')
    plt.xlabel('Cache level')
    plt.ylabel('Time (secs)')
    plt.legend()
    plt.savefig(img_path)
    plt.close()


def main():
    run_torch()


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(1147481649)

    reps = 500
    b, d = 512, 10

    t0, t1 = 0., 1.
    ts = torch.rand(size=(reps,)).numpy()

    main()

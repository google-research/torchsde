/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "utils.hpp"

#include <math.h>
#include <torch/torch.h>

#include <iomanip>
#include <random>
#include <sstream>

torch::Tensor brownian_bridge(float t, float t0, float t1, torch::Tensor w0,
                              torch::Tensor w1) {
  auto mean = ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0);
  auto std = std::sqrt((t1 - t) * (t - t0) / (t1 - t0));
  return mean + torch::randn_like(mean) * std;
}

std::string format_float(float t, int precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << t;
  return stream.str();
}

torch::Tensor brownian_bridge_with_seed(double t, double t0, double t1,
                                        torch::Tensor w0, torch::Tensor w1,
                                        std::uint64_t seed) {
  // TODO: Make this also work for CUDA. Related issue:
  // https://github.com/pytorch/pytorch/issues/35078.
  std::shared_ptr<at::CPUGenerator> curr = at::detail::createCPUGenerator(seed);
  torch::Tensor mean = ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0);
  double std = std::sqrt((t1 - t) * (t - t0) / (t1 - t0));
  torch::Tensor bridge_point = at::normal(mean, std, curr.get());
  curr.reset();
  return bridge_point;
}

torch::Tensor binary_search_with_seed(double t, double t0, double t1,
                                      torch::Tensor w0, torch::Tensor w1,
                                      std::uint64_t parent, double tol) {
  std::seed_seq seq({parent});
  std::vector<std::uint64_t> seeds(3);
  seq.generate(seeds.begin(), seeds.end());

  std::uint64_t seedv = seeds[0];
  std::uint64_t seedl = seeds[1];
  std::uint64_t seedr = seeds[2];

  auto t_mid = (t0 + t1) / 2;
  auto w_mid = brownian_bridge_with_seed(t_mid, t0, t1, w0, w1, seedv);

  while (std::abs(t - t_mid) > tol) {
    if (t < t_mid) {
      t1 = t_mid;
      w1 = w_mid;
      parent = seedl;
    } else {
      t0 = t_mid;
      w0 = w_mid;
      parent = seedr;
    }

    std::seed_seq seq({parent});
    seq.generate(seeds.begin(), seeds.end());
    seedv = seeds[0];
    seedl = seeds[1];
    seedr = seeds[2];

    t_mid = (t0 + t1) / 2;
    w_mid = brownian_bridge_with_seed(t_mid, t0, t1, w0, w1, seedv);
  }
  return w_mid;
}

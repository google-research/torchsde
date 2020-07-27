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

#ifndef BROWNIAN_TREE_HPP
#define BROWNIAN_TREE_HPP

#include <torch/torch.h>

#include <map>

class BrownianTree {
 private:
  int entropy;  // TODO: Use 64-bit based.
  double tol;
  int cache_depth;
  double safety;

  // TODO: Use std::vector-based cache for further speed up.
  std::map<double, torch::Tensor> cache;
  std::vector<std::uint64_t> seeds;

  std::map<double, torch::Tensor> cache_prev;
  std::map<double, torch::Tensor> cache_post;

 public:
  BrownianTree(double t0, torch::Tensor w0, double t1, int entropy, double tol,
               int cache_depth, double safety);

  BrownianTree(int entropy, double tol, double cache_depth, double safety,
               std::map<double, torch::Tensor> cache,
               std::map<double, torch::Tensor> cache_prev,
               std::map<double, torch::Tensor> cache_post,
               std::vector<std::uint64_t> seeds);

  torch::Tensor call(double t);

  std::string repr() const;

  std::vector<std::map<double, torch::Tensor>> get_cache() const;

  std::vector<std::uint64_t> get_seeds() const;

  double get_t0() const;

  double get_t1() const;

  torch::Tensor get_w0() const;

  torch::Tensor get_w1() const;
};

#endif

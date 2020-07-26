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
  double t0;
  double t1;
  torch::Tensor w0;
  torch::Tensor w1;
  int entropy;
  double tol;
  int cache_depth;
  double safety;

 public:
  BrownianTree(double t0, torch::Tensor w0, double t1, torch::Tensor w1,
               int entropy, double tol, int cache_depth, double safety);

  torch::Tensor call(double t);

  std::string repr() const;

  double get_t0() const;

  double get_t1() const;

  torch::Tensor get_w0() const;

  torch::Tensor get_w1() const;
};

#endif

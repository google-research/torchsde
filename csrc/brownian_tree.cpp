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

#include "brownian_tree.hpp"

#include <torch/torch.h>

#include "utils.hpp"

BrownianTree::BrownianTree(double t0, torch::Tensor w0, double t1,
                           torch::Tensor w1, int entropy, double tol,
                           int cache_depth, double safety) {
  this->t0 = t0;
  this->w0 = w0;
  this->t1 = t1;
  this->w1 = w1;
  this->entropy = entropy;
  this->tol = tol;
  this->cache_depth = cache_depth;
  this->safety = safety;
}

torch::Tensor BrownianTree::call(double t) {
  // TODO: Record last query depth.
  return binary_search_with_seed(t, t0, t1, w0, w1, entropy, tol);
}

std::string BrownianTree::repr() const {
  return "BrownianTree(t0=" + format_float(t0, 3) + ", " +
         "t1=" + format_float(t1, 3) + ", " +
         "entropy=" + std::to_string(entropy) + ", " +
         "tol=" + std::to_string(tol) + ")";
}

double BrownianTree::get_t0() const { return t0; }

double BrownianTree::get_t1() const { return t1; }

torch::Tensor BrownianTree::get_w0() const { return w0; }

torch::Tensor BrownianTree::get_w1() const { return w1; }

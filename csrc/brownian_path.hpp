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

#ifndef BROWNIAN_PATH_HPP
#define BROWNIAN_PATH_HPP

#include <torch/torch.h>

#include <map>

class BrownianPath {
 private:
  std::map<float, torch::Tensor> cache;

 public:
  BrownianPath(float t0, torch::Tensor w0);

  BrownianPath(std::map<float, torch::Tensor> data);

  torch::Tensor call(float t);

  void insert(float t, torch::Tensor w);

  std::string repr() const;

  std::map<float, torch::Tensor> get_cache() const;

  float get_t_head() const;

  float get_t_tail() const;

  torch::Tensor get_w_head() const;

  torch::Tensor get_w_tail() const;
};

#endif

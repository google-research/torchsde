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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <torch/torch.h>

torch::Tensor brownian_bridge(float t, float t0, float t1,
                              torch::Tensor const &w0, torch::Tensor const &w1);

std::string format_float(float t, int precision = 3);

torch::Tensor brownian_bridge_with_seed(double t, double t0, double t1,
                                        torch::Tensor const &w0,
                                        torch::Tensor const &w1,
                                        std::uint64_t seed);

// This function performs binary search with given global entropy and the
// left anchor (t0, w0) and right anchor (t1, w1). Returns the tensor at a time
// that is tol-close to query time t.
torch::Tensor binary_search_with_seed(double t, double t0, double t1,
                                      torch::Tensor w0, torch::Tensor w1,
                                      std::uint64_t root, double tol);

// This function populates the cache for BrownianTree. Two things are updated:
// 1) (t, w) pairs in the map, and 2) seeds for each small interval between two
// timestamps.
void populate_cache(double t0, torch::Tensor const &w0, double t1, int entropy,
                    int cache_depth, std::map<double, torch::Tensor> &cache,
                    std::vector<std::uint64_t> &seeds);

#endif

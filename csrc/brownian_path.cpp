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

#include <math.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>
#include <iterator>
#include <map>

#include "utils.hpp"

namespace py = pybind11;

class BrownianPath {
 private:
  std::map<float, torch::Tensor> cache;
  float t_head;
  float t_tail;
  torch::Tensor w_head;
  torch::Tensor w_tail;

 public:
  BrownianPath(float t0, torch::Tensor w0) {
    cache.insert(std::pair<float, torch::Tensor>(t0, w0));
    t_head = t0;
    t_tail = t0;
    w_head = w0;
    w_tail = w0;
  }

  torch::Tensor call(float t) {
    if (cache.find(t) != cache.end()) {
      return cache.at(t);
    } else if (t < t_head) {
      auto w = w_head + torch::randn_like(w_head) * sqrt(t_head - t);
      cache.insert(std::pair<float, torch::Tensor>(t, w));
      t_head = t;
      w_head = w;
      return w;
    } else if (t > t_tail) {
      auto w = w_tail + torch::randn_like(w_tail) * sqrt(t - t_tail);
      cache.insert(std::pair<float, torch::Tensor>(t, w));
      t_tail = t;
      w_tail = w;
      return w;
    } else {
      auto lo = cache.lower_bound(t);
      auto hi = lo--;
      auto w = brownian_bridge(t, lo->first, hi->first, lo->second, hi->second);
      cache.insert(std::pair<float, torch::Tensor>(t, w));
      return w;
    }
  }

  auto size() const { return w_head.sizes(); }

  auto device() const { return w_head.device(); }

  auto dtype() const { return w_head.dtype(); }

  auto get_cache() const { return cache; }

  auto repr() const {
    return "BrownianPath(t0=" + format_float(t_head) +
           ", t1=" + format_float(t_tail) + ")";
  }
};

PYBIND11_MODULE(_brownian_lib, m) {
  // TODO: Fix `device`, `dtype`, `to` functions.
  m.doc() = "Fast Brownian motion based on PyTorch C++ API.";
  py::class_<BrownianPath>(m, "BrownianPath")
      .def(py::init<float, torch::Tensor>(), py::arg("t0"), py::arg("w0"))
      .def("__call__", &BrownianPath::call, py::arg("t"))
      .def("get_cache", &BrownianPath::get_cache)
      .def("size", &BrownianPath::size)
      .def("__repr__", &BrownianPath::repr)
      .def_property_readonly("shape", &BrownianPath::size);
}

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "brownian_path.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_brownian_lib, m) {
  m.doc() = "Fast Brownian motion based on PyTorch C++ API.";
  py::class_<BrownianPath>(m, "BrownianPath")
      .def(py::init([](float t0, torch::Tensor w0) {
             return new BrownianPath(t0, w0);
           }),
           py::arg("t0"), py::arg("w0"))
      .def(py::init([](std::map<float, torch::Tensor> data) {
             return new BrownianPath(data);
           }),
           py::arg("data"))
      .def("__call__", &BrownianPath::call, py::arg("t"))
      .def("__repr__", &BrownianPath::repr)
      .def("insert", &BrownianPath::insert)
      .def("get_cache", &BrownianPath::get_cache)
      .def("get_t_head", &BrownianPath::get_t_head)
      .def("get_t_tail", &BrownianPath::get_t_tail)
      .def("get_w_head", &BrownianPath::get_w_head)
      .def("get_w_tail", &BrownianPath::get_w_tail);
}

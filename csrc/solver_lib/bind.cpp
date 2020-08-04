#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "srid2.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Solver acceleration functions in C++.";
  m.def("srid2_step", &srid2_step);
}

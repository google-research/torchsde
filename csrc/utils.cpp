#include "utils.hpp"

#include <math.h>
#include <torch/torch.h>

#include <iomanip>
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

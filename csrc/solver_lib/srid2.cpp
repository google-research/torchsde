#include "srid2.hpp"

#include <math.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <iterator>
#include <map>

using namespace std;
using namespace torch;

double A0[srid2_STAGES][srid2_STAGES] = {
    {0, 0, 0, 0},
    {1, 0, 0, 0},
    {1 / 4, 1 / 4, 0, 0},
    {0, 0, 0, 0}};
double A1[srid2_STAGES][srid2_STAGES] = {
    {0, 0, 0, 0},
    {1 / 4, 0, 0, 0},
    {1, 0, 0, 0},
    {0, 0, 1 / 4, 0}};

double B0[srid2_STAGES][srid2_STAGES] = {
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {1, 1 / 2, 0, 0},
    {0, 0, 0, 0}};
double B1[srid2_STAGES][srid2_STAGES] = {
    {0, 0, 0, 0},
    {-1 / 2, 0, 0, 0},
    {1, 0, 0, 0},
    {2, -1, 1 / 2, 0}};

double C0[srid2_STAGES]{0, 1, 1 / 2, 0};
double C1[srid2_STAGES]{0, 1 / 4, 1, 1 / 4};

double alpha[srid2_STAGES]{1 / 6, 1 / 6, 2 / 3, 0};
double beta1[srid2_STAGES]{-1, 4 / 3, 2 / 3, 0};
double beta2[srid2_STAGES]{1, -4 / 3, 1 / 3, 0};
double beta3[srid2_STAGES]{2, -4 / 3, -2 / 3, 0};
double beta4[srid2_STAGES]{-2, 5 / 3, -2 / 3, 1};

vector<Tensor> srid2_step(
    function<vector<Tensor>(double, vector<Tensor>)> const &f,
    function<vector<Tensor>(double, vector<Tensor>)> const &g,
    double t0,
    double dt,
    double sqrt_dt,
    vector<Tensor> const &y0,
    vector<Tensor> const &I_k,
    vector<Tensor> const &I_kk,
    vector<Tensor> const &I_k0,
    vector<Tensor> const &I_kkk) {
  auto y1 = y0;

  vector<vector<Tensor>> H0;
  vector<vector<Tensor>> H1;

  for (unsigned long s = 0; s < srid2_STAGES; s++) {
    auto H0s = y0;
    auto H1s = y0;

    vector<Tensor> f_eval;
    vector<Tensor> g_eval;

    for (unsigned long j = 0; j < s; j++) {
      f_eval = f(t0 + C0[j] * dt, H0[j]);
      g_eval = g(t0 + C1[j] * dt, H1[j]);

      for (unsigned long k = 0; k < y0.size(); k++) {
        H0s[k] = H0s[k] + A0[s][j] * f_eval[k] * dt + B0[s][j] * g_eval[k] * I_k0[k] / dt;
        H1s[k] = H1s[k] + A1[s][j] * f_eval[k] * dt + B1[s][j] * g_eval[k] * sqrt_dt;
      }
    }
    H0.push_back(H0s);
    H1.push_back(H1s);

    f_eval = f(t0 + C0[s] * dt, H0s);
    g_eval = g(t0 + C1[s] * dt, H1s);
    for (unsigned long k = 0; k < y0.size(); k++) {
      auto g_weight = beta1[s] * I_k[k] + beta2[s] * I_kk[k] / sqrt_dt + beta3[s] * I_k0[k] / dt + beta4[s] * I_kkk[k] / dt;
      y1[k] = y1[k] + alpha[s] * f_eval[k] * dt + g_weight * g_eval[k];
    }
  }
  return y1;
}

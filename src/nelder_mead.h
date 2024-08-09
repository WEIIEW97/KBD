/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <numeric>
#include <vector>
#include <functional>
#include <algorithm>
#include <type_traits>

namespace kbd {

  template <typename Scalar, class T_func, int Dim = Eigen::Dynamic>
  class NelderMeadOptimizer {
  public:
    static_assert(
        std::is_base_of<Eigen::MatrixBase<std::decay_t<T_func>>, T_func>::value,
        "T_func must be an Eigen::Vector type.");

    NelderMeadOptimizer(std::function<Scalar(const T_func&)> func,
                        Scalar alpha = 1.0, Scalar gamma = 2.0,
                        Scalar rho = 0.5, Scalar sigma = 0.5,
                        int max_iter = 10000, Scalar tol = 1e-6)
        : obj_func_(func), alpha_(alpha), gamma_(gamma), rho_(rho),
          sigma_(sigma), max_iter_(max_iter), tol_(tol) {}

    void set_init_simplex(const T_func& initial_guess) {
      int n = initial_guess.size();
      simplex_.resize(n + 1);
      for (int i = 0; i <= n; ++i) {
        simplex_[i] = initial_guess;
        if (i < n) {
          simplex_[i](i) +=
              (initial_guess(i) != 0) ? 0.05 * initial_guess(i) : 0.00025;
        }
      }
    }

    void set_bounds(const T_func& lower_bounds, const T_func& upper_bounds) {
      lower_bounds_ = std::move(lower_bounds);
      upper_bounds_ = std::move(upper_bounds);
      is_bounded_ = true;
    }

    T_func optimize() {
      auto n = simplex_[0].size();
      std::vector<Scalar> f_values(simplex_.size());

      cost_iter(f_values);

      for (int iter = 0; iter < max_iter_; ++iter) {
        std::sort(simplex_.begin(), simplex_.end(),
                  [&](const T_func& a, const T_func& b) {
                    return obj_func_(a) < obj_func_(b);
                  });

        cost_iter(f_values);

        if (check_convergence(f_values)) {
          fmt::print("Convergence achieved after {} iterations.\n", iter);
          break;
        }

        // Calculate centroid of all but worst point
        T_func centroid = T_func::Zero(n);
        for (int i = 0; i < simplex_.size() - 1; ++i) {
          centroid += simplex_[i];
        }
        centroid /= (simplex_.size() - 1);

        // Reflection
        T_func worst = simplex_.back();
        T_func reflected;
        if (!is_bounded_) {
          reflected = centroid + alpha_ * (centroid - worst);
        } else {
          reflected = clamp(centroid + alpha_ * (centroid - worst));
        }
        Scalar reflected_val = obj_func_(reflected);

        if (reflected_val < obj_func_(simplex_[0])) {
          // Expansion
          T_func expanded;
          if (!is_bounded_) {
            expanded = centroid + gamma_ * (reflected - centroid);
          } else {
            expanded = clamp(centroid + gamma_ * (reflected - centroid));
          }
          if (obj_func_(expanded) < reflected_val) {
            simplex_.back() = expanded;
          } else {
            simplex_.back() = reflected;
          }
        } else if (reflected_val < obj_func_(worst)) {
          // accept reflection
          simplex_.back() = reflected;
        } else {
          // contraction
          T_func contracted;
          if (!is_bounded_) {
            contracted = centroid + rho_ * (worst - centroid);
          } else {
            contracted = clamp(centroid + rho_ * (worst - centroid));
          }
          if (obj_func_(contracted) < obj_func_(worst)) {
            simplex_.back() = contracted;
          } else {
            // shrink
            if (!is_bounded_) {
              for (int i = 1; i < simplex_.size(); ++i) {
                simplex_[i] =
                    simplex_[0] + sigma_ * (simplex_[i] - simplex_[0]);
              }
            } else {
              for (int i = 1; i < simplex_.size(); ++i) {
                simplex_[i] =
                    clamp(simplex_[i] = simplex_[0] +
                                        sigma_ * (simplex_[i] - simplex_[0]));
              }
            }
          }
        }
      }
      return simplex_[0];
    }

  private:
    void cost_iter(std::vector<Scalar>& f_values) {
      for (int i = 0; i < simplex_.size(); ++i) {
        f_values[i] = obj_func_(simplex_[i]);
      }
    }

    bool check_convergence(const std::vector<Scalar>& f_values) {
      if (f_values.empty())
        return true;

      Scalar mu = std::accumulate(f_values.begin(), f_values.end(), Scalar(0)) /
                  f_values.size();
      Scalar sq_sum = std::inner_product(f_values.begin(), f_values.end(),
                                         f_values.begin(), Scalar(0));
      Scalar sigma = std::sqrt(sq_sum / f_values.size() - mu * mu);
      return sigma < tol_;
    };

    T_func clamp(const T_func& v) {
      return v.cwiseMax(lower_bounds_).cwiseMin(upper_bounds_);
    }

  private:
    std::vector<T_func> simplex_;
    Scalar alpha_, gamma_, rho_, sigma_;
    int max_iter_;
    Scalar tol_; // Tolerance for stopping criterion
    std::function<Scalar(const T_func&)> obj_func_;
    T_func lower_bounds_, upper_bounds_;
    bool is_bounded_ = false;
  };
} // namespace kbd
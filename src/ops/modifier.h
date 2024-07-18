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

#include "../utils.h"
#include "../array.h"

#include <Eigen/Dense>
#include <array>
#include <fmt/core.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace kbd {
  namespace ops {

    template <typename Derived>
    auto safe_divide(double numerator,
                     const Eigen::ArrayBase<Derived>& denominator) {
      // Using unaryExpr to safely divide with a lambda function
      return denominator.unaryExpr([numerator](double denom) -> double {
        return denom != 0
                   ? numerator / denom
                   : 0; // Return 0 or some other defined value if denom is 0
      });
    }

    template <typename Derived>
    ndArray<uint16_t>
    modify_linear(const ndArray<Derived>& m, double focal, double baseline,
                  const Eigen::Matrix<double, 5, 5>& param_matrix,
                  const std::array<int, 2>& disjoint_depth_range,
                  double compensate_dist, double scaling_factor) {
      double fb = focal * baseline;
      ndArray<double> out(m.rows(), m.cols());
      out.setZero();

      double lb = static_cast<double>(disjoint_depth_range[0]);
      double ub = static_cast<double>(disjoint_depth_range[1]);

      auto m_double = m.template cast<double>().array();
      auto d_double = safe_divide(fb, m_double);

      out =
          (m_double >= 0 && m_double < lb - compensate_dist)
              .select(m_double, 0.0) +
          (m_double >= lb - compensate_dist && m_double < lb)
              .select(fb / (param_matrix(1, 3) * d_double + param_matrix(1, 4)),
                      0.0) +
          (m_double >= lb && m_double < ub)
              .select(param_matrix(2, 0) * fb /
                              (d_double + param_matrix(2, 1)) +
                          param_matrix(2, 2),
                      0.0) +
          (m_double >= ub && m_double < ub + compensate_dist * scaling_factor)
              .select(fb / (param_matrix(3, 3) * d_double + param_matrix(3, 4)),
                      0.0) +
          (m_double >= ub + compensate_dist * scaling_factor)
              .select(m_double, 0.0);

      // Convert and clamp the final results to uint16_t using unaryExpr
      auto clamp_cast = [](double v) {
        return static_cast<uint16_t>(std::clamp(v, 0.0, 65535.0));
      };
      return out.unaryExpr(clamp_cast);
    }

    template <typename Derived>
    void apply_transformation_linear(
        const std::string& path,
        const Eigen::Matrix<double, 5, 5>& params_matrix, double focal,
        double baseline, const std::array<int, 2>& disjoint_depth_range,
        double compensate_dist, double scaling_factor, int H, int W,
        const std::string& subfix) {
      auto folders = retrieve_folder_names(path);

      for (int k = 0; k < folders.size(); ++k) {
        const auto& folder = folders[k];
        auto full_path = path + "/" + folder + "/" + subfix;
        // auto full_path = fs::path(path) / folder / subfix;
        for (const auto& entry : fs::directory_iterator(full_path)) {
          auto p = entry.path().string();
          auto raw = load_raw<uint16_t>(p, H, W);
          auto depth = modify_linear<uint16_t>(
              raw, focal, baseline, params_matrix, disjoint_depth_range,
              compensate_dist, scaling_factor);
          save_raw<uint16_t>(p, depth);
        }
      }
      fmt::print("Transformating data done ...\n");
    }
    
    void parallel_copy(const std::string& src, const std::string& dst,
                       const Config& configs);
    void parallel_transform(const std::string& path,
                            const Eigen::Matrix<double, 5, 5>& params_matrix,
                            double focal, double baseline,
                            const Config& configs,
                            const JointSmoothArguments& arguments);
  } // namespace ops
} // namespace kbd
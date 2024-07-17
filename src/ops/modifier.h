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
#include "../eigen_io.h"

#include <Eigen/Dense>
#include <array>
#include <fmt/core.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace kbd {
  namespace ops {
    template <typename Derived>
    ndArray<Derived>
    modify_linear(const ndArray<Derived>& m, double focal, double baseline,
                  const Eigen::Matrix<double, 5, 5>& param_matrix,
                  const std::array<int, 2>& disjoint_depth_range,
                  double compensate_dist, double scaling_factor) {
      auto fb = focal * baseline;
      ndArray<double> out(m.rows(), m.cols());
      out.setZero();

      auto lb = disjoint_depth_range[0];
      auto ub = disjoint_depth_range[1];

      auto mask0 = (m.array() >= 0) && (m.array() < lb - compensate_dist);
      auto mask1 = (m.array() >= lb - compensate_dist) && (m.array() < lb);
      auto mask2 = (m.array() >= lb) && (m.array() < ub);
      auto mask3 = (m.array() >= ub) &&
                   (m.array() < ub + compensate_dist * scaling_factor);
      auto mask4 = (m.array() >= ub + compensate_dist * scaling_factor);

      out =
          ((mask0).template cast<double>() * m).matrix() +
          ((mask1).template cast<double>() *
           (fb / (param_matrix(1, 3) * (fb / m.array()) + param_matrix(1, 4))))
              .matrix() +
          ((mask2).template cast<double>() *
           (param_matrix(2, 0) * fb / ((fb / m.array()) + param_matrix(2, 1)) +
            param_matrix(2, 2)))
              .matrix() +
          ((mask3).template cast<double>() *
           (fb / (param_matrix(3, 3) * (fb / m.array()) + param_matrix(3, 4))))
              .matrix() +
          ((mask4).template cast<double>() * m).matrix();

      return out.cast<Derived>();
    }

    template <typename Derived>
    void apply_transformation_linear(
        const std::string& path,
        const Eigen::Matrix<double, 5, 5>& params_matrix, double focal,
        double baseline, const std::array<int, 2>& disjoint_depth_range,
        double compensate_dist, double scaling_factor, int H, int W) {
      auto folders = retrieve_folder_names(path);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
      for (int k = 0; k < folders.size(); ++k) {
        const auto& folder = folders[k];
        for (const auto& entry : fs::directory_iterator(folder)) {
          auto p = entry.path();
          auto raw = load_raw<uint16_t>(p, H, W);
          auto depth = modify_linear<uint16_t>(
              raw, focal, baseline, params_matrix, disjoint_depth_range,
              compensate_dist, scaling_factor);
          write_binary(p.string(), depth);
        }
      }
      fmt::print("Transformating data done ...");
    }

  } // namespace ops
} // namespace kbd
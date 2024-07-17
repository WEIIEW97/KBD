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

#include "config.h"
#include "table.h"
#include <string>
#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace kbd {

  class LinearWorkflow {
  public:
    LinearWorkflow() {}
    ~LinearWorkflow() = default;

    void preprocessing(const std::string& file_path,
                       const std::string& csv_path, const Config& config,
                       const JointSmoothArguments& args);
    void optimize();
    void extend_matrix();
    std::tuple<Eigen::Array<uint16_t, Eigen::Dynamic, 1>,
               Eigen::Matrix<double, 5, 5>>
    pivot();
    std::tuple<std::map<double, double>, double> eval(const Config& config);

  private:
    double focal_, baseline_, cd_, sf_;
    bool apply_global_;
    Eigen::ArrayXd gt_double_, est_double_;
    Eigen::Vector2d lm1_, lm2_;
    Eigen::Vector3d kbd_res_;
    std::array<int, 2> disjoint_depth_range_ = {0};
    uint16_t disp_val_max_uint16_;

  public:
    std::shared_ptr<arrow::Table> trimmed_df_;
    Eigen::Matrix<double, 5, 5> full_kbd_params5x5_;
  };
} // namespace kbd
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
#include "array.h"
#include "optimizer.h"

#include <string>
#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace kbd {

  class LinearWorkflow {
  public:
    LinearWorkflow() = default;
    ~LinearWorkflow() = default;

    void preprocessing(const std::string& file_path,
                       const std::string& csv_path, const Config& config,
                       const JointSmoothArguments& args);
    void optimize(OptimizerDiffType diff_type = OptimizerDiffType::NELDER_MEAD);
    void
    line_search(const std::array<int, 2>& search_range,
                OptimizerDiffType diff_type = OptimizerDiffType::NELDER_MEAD);
    struct grid_search {
      explicit grid_search(
          LinearWorkflow* lwf,
          OptimizerDiffType diff_type = OptimizerDiffType::NELDER_MEAD)
          : lwf_(lwf), diff_type_(diff_type) {}

      std::tuple<double, Eigen::Matrix<double, 5, 5>, Eigen::Vector2d,
                 Eigen::Vector3d, Eigen::Vector2d>
      eval_params(int range_start, double compensate_dist);
      void optimize_params(const std::array<int, 2>& search_range,
                           const std::array<double, 2>& cd_range,
                           int max_iter = 1000, double tol = 1e-6);
      std::tuple<Eigen::Matrix<double, 5, 5>, int, double> get_results();

      double objective(const Eigen::Vector2d& params);

    public:
      Eigen::Vector2d optim_res_;

    private:
      LinearWorkflow* lwf_;
      OptimizerDiffType diff_type_;
    };
    void extend_matrix();
    Eigen::Matrix<double, 5, 5> extend_matrix(const Eigen::Vector2d& lm1,
                                              const Eigen::Vector3d& kbd,
                                              const Eigen::Vector2d& lm2);
    std::tuple<Eigen::Array<uint16_t, Eigen::Dynamic, 1>,
               Eigen::Matrix<double, 5, 5>>
    pivot();
    std::tuple<Eigen::Array<uint16_t, Eigen::Dynamic, 1>,
               Eigen::Matrix<double, 5, 5>>
    pivot(const Eigen::Matrix<double, 5, 5>& m, const std::array<int, 2>& rng,
          double cd);
    std::tuple<Eigen::Array<uint16_t, Eigen::Dynamic, 1>,
               Eigen::Matrix<double, 5, 5>>
    export_default() const;
    std::tuple<std::map<double, double>, double> eval();
    bool pass_or_not();
    bool ratio_evaluate(double alpha = 0.5, int min_offset = 500);
    bool first_check(double max_thr = 0.07, double mean_thr = 0.05);
    bool final_check(const Eigen::Matrix<double, 5, 5>& pm,
                     const std::array<int, 2>& range, double cd,
                     double weights_factor = 3.0);
    double get_focal_val() const;
    double get_baseline_val() const;

  private:
    std::tuple<double, Eigen::Vector<double, 6>>
    evaluate_target(const Eigen::Matrix<double, 5, 5>& param_matrix,
                    const std::array<int, 2>& rg);
    void lazy_compute_ref_z();

  private:
    double focal_{}, baseline_{}, cd_{}, sf_{};
    bool apply_global_ = false;
    Eigen::ArrayXd gt_double_, est_double_, error_double_;
    Eigen::Vector2d lm1_, lm2_;
    Eigen::Vector3d kbd_res_;
    std::array<int, 2> disjoint_depth_range_ = {0};
    std::array<int, 2> best_range_ = {0};
    Eigen::Matrix<double, 5, 5> best_pm_;
    Eigen::Vector<double, 6> best_z_error_rate_;
    ndArray<double> ref_z_;
    uint16_t disp_val_max_uint16_{};
    int step_ = 50; // can be modified accordingly
    std::array<int, 6> metric_points_ = JointSmoothArguments().metric_points;
    Config config_;
    JointSmoothArguments args_;

  public:
    std::shared_ptr<arrow::Table> trimmed_df_;
    Eigen::Matrix<double, 5, 5> full_kbd_params5x5_;
    friend struct grid_search;
    bool final_pass_ = false;
  };
} // namespace kbd
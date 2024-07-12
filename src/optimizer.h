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

#include "model.h"

#include <ceres/ceres.h>
#include <vector>
#include <array>

namespace kbd {

  class CeresAutoDiffOptimizer {
  public:
    CeresAutoDiffOptimizer(Eigen::ArrayXd gt, Eigen::ArrayXd est, double focal,
                           double baseline)
        : gt_(std::move(gt)), est_(std::move(est)), focal_(focal),
          baseline_(baseline) {
      initial_params_ = {1.0, 0.01, 10.0};
      options_.max_num_iterations = 1000;
      options_.linear_solver_type = ceres::DENSE_QR;
    }

    struct CostFunctor {
      explicit CostFunctor(CeresAutoDiffOptimizer* optimizer)
          : optimizer_(optimizer) {}

      template <typename T>
      bool operator()(const T* const params, T* residual) const {
        T k = params[0];
        T delta = params[1];
        T b = params[2];

        // Create an Eigen::Array<T, Eigen::Dynamic, 1> from optimizer_->est_ by
        // explicit casting
        Eigen::Array<T, Eigen::Dynamic, 1> disp =
            optimizer_->est_.template cast<T>();

        // Call the templated basic_model function
        Eigen::Array<T, Eigen::Dynamic, 1> pred = basic_model(
            disp, T(optimizer_->focal_), T(optimizer_->baseline_), k, delta, b);
        Eigen::Array<T, Eigen::Dynamic, 1> residuals =
            optimizer_->gt_.template cast<T>() - pred;
        T mse = residuals.square().mean();

        residual[0] = mse;
        return true;
      }

      CeresAutoDiffOptimizer* optimizer_;
    };

    void set_ceres_options(int max_num_iterations,
                           ceres::LinearSolverType linear_solver_type) {
      options_.max_num_iterations = max_num_iterations;
      options_.linear_solver_type = linear_solver_type;
    };

    void set_opt_initial_params(const std::array<double, 3>& initial_params) {
      initial_params_ = initial_params;
    }

    Eigen::Vector3d run() {
      ceres::Problem problem;
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(
              new CostFunctor(this)),
          nullptr, initial_params_.data());

      ceres::Solver::Summary summary;
      ceres::Solve(options_, &problem, &summary);

      std::cout << summary.FullReport() << std::endl;
      std::cout << "Optimized Parameters: k = " << initial_params_[0]
                << ", delta = " << initial_params_[1]
                << ", b = " << initial_params_[2] << std::endl;

      Eigen::Vector3d kbd_params = {initial_params_[0], initial_params_[1],
                                    initial_params_[2]};
      return kbd_params;
    }

  private:
    Eigen::ArrayXd gt_;
    Eigen::ArrayXd est_;
    double focal_;
    double baseline_;
    std::array<double, 3> initial_params_;
    ceres::Solver::Options options_;
  };

  class JointLinearSmoothingOptimizer {
  public:
    JointLinearSmoothingOptimizer(
        const Eigen::ArrayXd& gt, const Eigen::ArrayXd& est, double focal,
        double baseline, const std::array<int, 2>& disjoint_depth_range,
        double compensate_dist = 200, double scaling_factor = 10,
        bool apply_global = false)
        : gt_(gt), est_(est), focal_(focal), baseline_(baseline),
          disjoint_depth_range_(disjoint_depth_range),
          compensate_dist_(compensate_dist), scaling_factor_(scaling_factor),
          apply_global_(apply_global) {
      fb_ = focal * baseline;
      initial_params_ = {1.0, 0.01, 10.0};
    }

    Eigen::VectorXd segment() {
      Eigen::Array<bool, Eigen::Dynamic, 1> mask =
          (gt_ > disjoint_depth_range_[0] && gt_ < disjoint_depth_range_[1]);
      if (apply_global_) {
        kbd_x_ = est_;
        kbd_y_ = gt_;
      } else {
        kbd_x_ = est_.select(mask, est_.setConstant(0));
        kbd_y_ = est_.select(mask, gt_.setConstant(0));
      }

      if (kbd_x_.size() == 0 || kbd_y_.size() == 0) {
        return Eigen::VectorXd();
      }

      CeresAutoDiffOptimizer kbd_base_optimizer(kbd_y_, kbd_x_, focal_,
                                                baseline_);
      auto res = kbd_base_optimizer.run();
      return res;
    }

  private:
    Eigen::ArrayXd gt_;
    Eigen::ArrayXd est_;
    double focal_;
    double baseline_;
    std::array<int, 2> disjoint_depth_range_;
    double compensate_dist_;
    double scaling_factor_;
    double fb_;
    bool apply_global_;
    std::vector<double> initial_params_;
    Eigen::ArrayXd kbd_x_;
    Eigen::ArrayXd kbd_y_;
  };
} // namespace kbd
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

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include "../src/optimizer.h"

//// Define the basic model function
//template <typename T>
//Eigen::Array<T, Eigen::Dynamic, 1> basic_model(const Eigen::Array<T, Eigen::Dynamic, 1>& disp, T focal, T baseline, T k, T delta, T b) {
//  return k * focal * baseline / (disp + delta) + b;
//}
//
//class CeresAutoDiffOptimizer {
//public:
//  CeresAutoDiffOptimizer(Eigen::ArrayXd  gt, Eigen::ArrayXd  est, double focal, double baseline)
//      : gt_(std::move(gt)), est_(std::move(est)), focal_(focal), baseline_(baseline) {
//    initial_params_ = {1.0, 0.01, 10.0};
//  }
//
//  struct CostFunctor {
//    explicit CostFunctor(CeresAutoDiffOptimizer* optimizer) : optimizer_(optimizer) {}
//
//    template <typename T>
//    bool operator()(const T* const params, T* residual) const {
//      T k = params[0];
//      T delta = params[1];
//      T b = params[2];
//
//      // Create an Eigen::Array<T, Eigen::Dynamic, 1> from optimizer_->est_ by explicit casting
//      Eigen::Array<T, Eigen::Dynamic, 1> disp = optimizer_->est_.template cast<T>();
//
//      // Call the templated basic_model function
//      Eigen::Array<T, Eigen::Dynamic, 1> pred = basic_model(disp, T(optimizer_->focal_), T(optimizer_->baseline_), k, delta, b);
//      Eigen::Array<T, Eigen::Dynamic, 1> residuals = optimizer_->gt_.template cast<T>() - pred;
//      T mse = residuals.square().mean();
//
//      residual[0] = mse;
//      return true;
//    }
//
//    CeresAutoDiffOptimizer* optimizer_;
//  };
//
//  void run() {
//    ceres::Problem problem;
//    problem.AddResidualBlock(
//        new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(this)),
//        nullptr, initial_params_.data()
//    );
//
//    ceres::Solver::Options options;
//    options.max_num_iterations = 1000;
//    options.linear_solver_type = ceres::DENSE_QR;
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//
//    std::cout << summary.FullReport() << std::endl;
//    std::cout << "Optimized Parameters: k = " << initial_params_[0]
//              << ", delta = " << initial_params_[1]
//              << ", b = " << initial_params_[2] << std::endl;
//  }
//
//private:
//  Eigen::ArrayXd gt_;
//  Eigen::ArrayXd est_;
//  double focal_;
//  double baseline_;
//  std::vector<double> initial_params_;
//};

int main() {
  // Example data (replace with actual data)
  Eigen::ArrayXd gt(3);
  gt << 1.0, 2.0, 3.0;
  Eigen::ArrayXd est(3);
  est << 1.1, 2.1, 3.1;
  double focal = 1.0;
  double baseline = 1.0;

  // Create and run the optimizer
  kbd::CeresAutoDiffOptimizer optimizer(gt, est, focal, baseline);
  auto kbd_params = optimizer.run();

  for(const auto& p:kbd_params) {
    std::cout << p << "\n";
  }
  std::cout << gt << std::endl;
  return 0;
}

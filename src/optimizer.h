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
#include "array.h"
#include "model.h"
#include <vector>

namespace kbd {

  class CeresAutoDiffOptimizer {
  public:
    CeresAutoDiffOptimizer(const Array<double>& gt, const Array<double>& est,
                           double focal, double baseline)
        : gt_(gt), est_(est), focal_(focal), baseline_(baseline) {
      initial_params_ = {1.0, 0.01, 10.0};
    }

    struct CostFunctor {
      CostFunctor(CeresAutoDiffOptimizer* optimizer) : optimizer_(optimizer) {}

      template <typename T>
      bool operator()(const T* const params, T* residual) const {
        T k = params[0];
        T delta = params[1];
        T b = params[2];

        Eigen::Array<T, Eigen::Dynamic, 1> pred = basic_model(optimizer_->est_.cast<T>(), optimizer_->focal_, optimizer_->baseline_, k, delta, b);
        Eigen::Array<T, Eigen::Dynamic, 1> residuals = optimizer_->gt_.cast<T>() - pred;
        T mse = residuals.square().mean();

        residual[0] = mse;
        return true;
      }

      CeresAutoDiffOptimizer* optimizer_;
    };

    void run() {
      ceres::Problem problem;
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(
              new CostFunctor(this)),
          nullptr, initial_params_.data());

      ceres::Solver::Options options;
      options.max_num_iterations = 1000;
      options.linear_solver_type = ceres::DENSE_QR;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      std::cout << summary.FullReport() << std::endl;
      std::cout << "Optimized Parameters: k = " << initial_params_[0]
                << ", delta = " << initial_params_[1]
                << ", b = " << initial_params_[2] << std::endl;
    }

  private:
    Eigen::ArrayXd gt_;
    Eigen::ArrayXd est_;
    double focal_;
    double baseline_;
    std::vector<double> initial_params_;
  };
} // namespace kbd
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
#include "eigen_utils.h"
#include "linear_reg.h"
#include "format.h"

#include <ceres/ceres.h>
#include <vector>
#include <array>
#include <memory>
#include <functional>

namespace kbd {
  enum OptimizerDiffType {
    AUTO_DIFF,
    NUMERICAL_DIFF,
    NELDER_MEAD,
  };

  class CeresBaseOptimizer {
  public:
    virtual Eigen::Vector3d run() = 0;
    virtual ~CeresBaseOptimizer() {}
  };

  class CeresAutoDiffOptimizer : public CeresBaseOptimizer {
  public:
    CeresAutoDiffOptimizer(Eigen::ArrayXd gt, Eigen::ArrayXd est, double focal,
                           double baseline)
        : gt_(std::move(gt)), est_(std::move(est)), focal_(focal),
          baseline_(baseline) {
      initial_params_ = {1.0, 0.01, 10.0};
      options_.max_num_iterations = 1000;
      options_.linear_solver_type = ceres::DENSE_QR;
      options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options_.num_threads = 4;
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

    void set_opt_num_of_threads(int n_threads) {
      options_.num_threads = n_threads;
    }

    Eigen::Vector3d run() override {
      ceres::Problem problem;
      auto loss_function = ceres::HuberLoss(1.0);
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(
              new CostFunctor(this)),
          &loss_function, initial_params_.data());

      ceres::Solver::Summary summary;
      ceres::Solve(options_, &problem, &summary);

      fmt::print("{}\n", summary.FullReport());
      fmt::print("Optimized Parameters: k = {}, delta = {}, b = {}\n",
                 initial_params_[0], initial_params_[1], initial_params_[2]);

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

  class CeresNumericalDiffOptimizer : public CeresBaseOptimizer {
  public:
    CeresNumericalDiffOptimizer(Eigen::ArrayXd gt, Eigen::ArrayXd est,
                                double focal, double baseline)
        : gt_(std::move(gt)), est_(std::move(est)), focal_(focal),
          baseline_(baseline) {
      initial_params_ = {1.0, 0.01, 10.0}; // Example initial parameters
      options_.max_num_iterations = 1000;
      options_.linear_solver_type = ceres::DENSE_QR;
      options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options_.num_threads = 4;
    }

    // Cost functor that calculates residuals
    struct CostFunctor {
      explicit CostFunctor(const Eigen::ArrayXd& gt, const Eigen::ArrayXd& est,
                           double focal, double baseline)
          : gt_(gt), est_(est), focal_(focal), baseline_(baseline) {}

      template <typename T>
      bool operator()(const T* const params, T* residual) const {
        T k = params[0];
        T delta = params[1];
        T b = params[2];

        Eigen::Array<T, Eigen::Dynamic, 1> pred = basic_model(
            est_.template cast<T>(), T(focal_), T(baseline_), k, delta, b);
        Eigen::Array<T, Eigen::Dynamic, 1> residuals =
            gt_.template cast<T>() - pred;
        residual[0] = residuals.square().mean();
        return true;
      }

    private:
      Eigen::ArrayXd gt_;
      Eigen::ArrayXd est_;
      double focal_;
      double baseline_;
    };

    void set_ceres_options(int max_num_iterations,
                           ceres::LinearSolverType linear_solver_type) {
      options_.max_num_iterations = max_num_iterations;
      options_.linear_solver_type = linear_solver_type;
    }

    void set_opt_initial_params(const std::array<double, 3>& initial_params) {
      initial_params_ = initial_params;
    }

    void set_opt_num_of_threads(int n_threads) {
      options_.num_threads = n_threads;
    }

    Eigen::Vector3d run() override {
      ceres::Problem problem;
      auto loss_function =
          new ceres::HuberLoss(1.0); // Huber loss to manage outliers
      ceres::CostFunction* cost_function =
          new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 1, 3>(
              new CostFunctor(gt_, est_, focal_, baseline_),
              ceres::TAKE_OWNERSHIP);

      problem.AddResidualBlock(cost_function, loss_function,
                               initial_params_.data());

      ceres::Solver::Summary summary;
      ceres::Solve(options_, &problem, &summary);

      fmt::print("{}\n", summary.FullReport());
      fmt::print("Optimized Parameters: k = {}, delta = {}, b = {}\n",
                 initial_params_[0], initial_params_[1], initial_params_[2]);

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

  template <typename T, int Dimension = Eigen::Dynamic>
  class NelderMeadOptimizer {
  public:
    using Vector = Eigen::Matrix<T, Dimension, 1>;
    using Matrix = Eigen::Matrix<T, Dimension, Dimension>;

    NelderMeadOptimizer(std::function<T(const Vector&)> objective_func,
                        const Vector& initial_guess, T alpha = 1.0,
                        T gamma = 2.0, T rho = 0.5, T sigma = 0.5)
        : objective_func_(objective_func), alpha_(alpha), rho_(rho),
          gamma_(gamma), sigma_(sigma) {
      auto n = initial_guess.size();
      simplex_.resize(Dimension, n + 1);
      // Initialize simplex
      for (int i = 0; i < n; ++i) {
        simplex_.col(i) = initial_guess;
        simplex_(i, i) +=
            (initial_guess(i) != 0) ? 0.05 * initial_guess(i) : 0.00025;
      }
      simplex_.col(n) = initial_guess; // last column is the initial guess

      fvalues_.resize(n + 1);
      for (int i = 0; i <= n; ++i) {
        fvalues_(i) = objective_func_(simplex_.col(i));
      }
    }

    Vector optimize(int max_iterations = 1000, T tol = 1e-6) {
      auto n = simplex_.rows();
      Vector centroid(n), reflected(n), expanded(n), contracted(n);
      T f_reflected, f_expanded, f_contracted;
      int iter = 0;

      while (iter < max_iterations) {
        sort_simplex();

        if (std::abs(fvalues_(n) - fvalues_(0)) < tol) {
          break;
        }

        // compute centroid of all points except the worst
        centroid = simplex_.leftCols(n).rowwise().mean();
        reflected = centroid + alpha_ * (centroid - simplex_.col(n));
        f_reflected = objective_func_(reflected);

        if (f_reflected < fvalues_(0)) {
          expanded = centroid + gamma_ * (reflected - centroid);
          f_expanded = objective_func_(expanded);
          if (f_expanded < f_reflected) {
            simplex_.col(n) = expanded;
            fvalues_(n) = f_expanded;
          } else {
            simplex_.col(n) = reflected;
            fvalues_(n) = f_reflected;
          }
        } else if (f_reflected < fvalues_(n - 1)) {
          simplex_.col(n) = reflected;
          fvalues_(n) = f_reflected;
        } else {
          if (f_reflected < fvalues_(n)) {
            contracted = centroid + rho_ * (reflected - centroid);
            f_contracted = objective_func_(contracted);
            if (f_contracted < f_reflected) {
              simplex_.col(n) = f_contracted;
              fvalues_(n) = f_contracted;
              continue;
            }
          } else {
            contracted = centroid + rho_ * (simplex_.col(n) - centroid);
            f_contracted = objective_func_(contracted);
            if (f_contracted < fvalues_(n)) {
              simplex_.col(n) = contracted;
              fvalues_(n) = f_contracted;
              continue;
            }
          }
          // shrink
          for (int i = 1; i <= n; ++i) {
            simplex_.col(i) =
                simplex_.col(0) + sigma_ * (sigma_.col(i) - simplex_.col(0));
            fvalues_(i) = objective_func_(simplex_.col(i));
          }
        }
        iter++;
      }

      return simplex_.col(0);
    }

  private:
    Matrix simplex_;
    Eigen::VectorXd fvalues_;
    T alpha_, gamma_, rho_, sigma_;
    std::function<T(const Vector&)> objective_func_;

  private:
    void sort_simplex() {
      std::vector<int> idx(fvalues_.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
                [&](int i, int j) { return fvalues_(i) < fvalues_(j); });

      Matrix sorted_simplex = simplex_;
      for (int i = 0; i < idx.size(); ++i) {
        simplex_.col(i) = sorted_simplex.col(idx[i]);
      }

      std::sort(fvalues_.data(), fvalues_.data() + fvalues_.size());
    }
  };

  class JointLinearSmoothingOptimizer {
  public:
    JointLinearSmoothingOptimizer(
        const Eigen::ArrayXd& gt, const Eigen::ArrayXd& est, double focal,
        double baseline, const std::array<int, 2>& disjoint_depth_range,
        double compensate_dist = 200, double scaling_factor = 10,
        bool apply_global = false, OptimizerDiffType diff_type = AUTO_DIFF)
        : gt_(gt), est_(est), focal_(focal), baseline_(baseline),
          disjoint_depth_range_(disjoint_depth_range),
          compensate_dist_(compensate_dist), scaling_factor_(scaling_factor),
          apply_global_(apply_global), diff_type_(diff_type) {
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
        kbd_x_ = mask_out_array<double, bool>(est_, mask);
        kbd_y_ = mask_out_array<double, bool>(gt_, mask);
      }

      if (kbd_x_.size() == 0 || kbd_y_.size() == 0) {
        return Eigen::VectorXd();
      }

      std::unique_ptr<CeresBaseOptimizer> kbd_base_optimizer;

      switch (diff_type_) {
      case AUTO_DIFF:
        kbd_base_optimizer = std::make_unique<CeresAutoDiffOptimizer>(
            kbd_y_, kbd_x_, focal_, baseline_);
      case NUMERICAL_DIFF:
        kbd_base_optimizer = std::make_unique<CeresNumericalDiffOptimizer>(
            kbd_y_, kbd_x_, focal_, baseline_);
      default:
        std::cerr << "unsupported differential type !" << std::endl;
      }

      auto res = kbd_base_optimizer->run();
      return res;
    }

    std::tuple<Eigen::Vector2d, Eigen::Vector3d, Eigen::Vector2d> run() {
      std::tuple<Eigen::Vector2d, Eigen::Vector3d, Eigen::Vector2d> params;
      auto kbd_res = segment();
      assert(kbd_res.size() == 3);
      double k, delta, b, x_min, x_max, y_hat_max, y_hat_min, x_hat_min,
          x_hat_max, pre_y, after_y, pre_x, after_x;

      k = kbd_res(0), delta = kbd_res(1), b = kbd_res(2);

      x_min = kbd_x_.minCoeff();
      x_max = kbd_x_.maxCoeff();

      y_hat_max = k * fb_ / (x_min + delta) + b;
      y_hat_min = k * fb_ / (x_max + delta) + b;

      x_hat_min = fb_ / y_hat_max;
      x_hat_max = fb_ / y_hat_min;

      pre_y = y_hat_min - compensate_dist_;
      after_y = y_hat_max + compensate_dist_ * scaling_factor_;

      pre_x = fb_ / pre_y;
      after_x = fb_ / after_y;

      Eigen::Vector2d x1(pre_x, x_max);
      Eigen::Vector2d y1(pre_x, x_hat_max);

      Eigen::Vector2d x2(x_min, after_x);
      Eigen::Vector2d y2(x_hat_min, after_x);

      auto lm1 = linear_regression<double>(x1, y1);
      auto lm2 = linear_regression<double>(x2, y2);

      return std::make_tuple(lm1, kbd_res, lm2);
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
    OptimizerDiffType diff_type_;
  };
} // namespace kbd
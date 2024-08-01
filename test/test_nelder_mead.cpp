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
#include <iostream>
#include <Eigen/Dense>
#include <numeric>
#include <vector>
#include <functional>
#include <cmath> // For std::pow and other math functions

#include "../src/optimizer.h"

using Vector = Eigen::VectorXd;

// Define the model and cost function
template <typename Scalar, int Dim = Eigen::Dynamic>
class NelderMeadOptimizer {
public:
  using Vec = Eigen::Matrix<Scalar, Dim, 1>;

  NelderMeadOptimizer(std::function<Scalar(const Vec&)> func,
                      Scalar alpha = 1.0, Scalar gamma = 2.0, Scalar rho = 0.5,
                      Scalar sigma = 0.5, int max_iter = 10000,
                      Scalar tol = 1e-6)
      : obj_func_(func), alpha_(alpha), gamma_(gamma), rho_(rho), sigma_(sigma),
        max_iter_(max_iter), tol_(tol) {}

  void set_init_simplex(const Vec& initial_guess) {
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

  Vec optimize() {
    auto n = simplex_[0].size();
    std::vector<Scalar> functionValues(simplex_.size());

    for (int i = 0; i < simplex_.size(); ++i) {
      functionValues[i] = obj_func_(simplex_[i]);
    }

    for (int iter = 0; iter < max_iter_; ++iter) {
      std::sort(simplex_.begin(), simplex_.end(),
                [&](const Vec& a, const Vec& b) {
                  return obj_func_(a) < obj_func_(b);
                });

      // Reevaluate function values after sorting
      for (int i = 0; i < simplex_.size(); ++i) {
        functionValues[i] = obj_func_(simplex_[i]);
      }

      // Check convergence criteria: standard deviation of function values
      Scalar meanValue = std::accumulate(functionValues.begin(),
                                         functionValues.end(), Scalar(0)) /
                         functionValues.size();
      Scalar sq_sum =
          std::inner_product(functionValues.begin(), functionValues.end(),
                             functionValues.begin(), Scalar(0));
      Scalar stdDev =
          std::sqrt(sq_sum / functionValues.size() - meanValue * meanValue);
      if (stdDev < tol_) {
        std::cout << "Convergence achieved after " << iter << " iterations."
                  << std::endl;
        break;
      }

      // Calculate centroid of all but worst point
      Vec centroid = Vec::Zero(n);
      for (int i = 0; i < simplex_.size() - 1; ++i) {
        centroid += simplex_[i];
      }
      centroid /= (simplex_.size() - 1);

      // Reflection
      Vec worst = simplex_.back();
      Vec reflected = centroid + alpha_ * (centroid - worst);
      Scalar reflected_val = obj_func_(reflected);

      if (reflected_val < obj_func_(simplex_[0])) {
        // Expansion
        Vec expanded = centroid + gamma_ * (reflected - centroid);
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
        Vec contracted = centroid + rho_ * (worst - centroid);
        if (obj_func_(contracted) < obj_func_(worst)) {
          simplex_.back() = contracted;
        } else {
          // shrink
          for (int i = 1; i < simplex_.size(); ++i) {
            simplex_[i] = simplex_[0] + sigma_ * (simplex_[i] - simplex_[0]);
          }
        }
      }
    }
    return simplex_[0];
  }

private:
  std::vector<Vec> simplex_;
  Scalar alpha_, gamma_, rho_, sigma_;
  int max_iter_;
  Scalar tol_; // Tolerance for stopping criterion
  std::function<Scalar(const Vec&)> obj_func_;
};

double costFunction(const Vector& params, const Eigen::ArrayXd& X,
                    const Eigen::ArrayXd& Y, double focal, double baseline) {
  double k = params(0);
  double delta = params(1);
  double b = params(2);

  // Calculate predicted Y_p based on the model
  Eigen::ArrayXd Y_p = k * focal * baseline / (X + delta) + b;

  // Compute the mean squared error (MSE) as the cost
  return (Y_p - Y).square().mean();
}

int main() {
  double focal = 1000.0; // Example value
  double baseline = 0.1; // Example value

  // Example data
  Eigen::ArrayXd X =
      Eigen::ArrayXd::LinSpaced(100, 1, 100); // Simulated distance measurements
  Eigen::ArrayXd Y = 1.0 / X.array(); // Simulated disparity measurements,
                                      // example inverse relationship

  // Initial parameters [k, delta, b]
  Vector initialGuess(3);
  initialGuess << 1.0, 0.1, 0.0; // Initial guesses for k, delta, and b

  // Create an instance of the optimizer with the cost function
    kbd::NelderMeadOptimizer<double> optimizer(
      std::bind(costFunction, std::placeholders::_1, X, Y, focal,
                baseline), // Bind X, Y, focal, and baseline to the function
      1.0, // alpha
      2.0, // gamma
      0.5, // rho
      0.5  // sigma
  );

  optimizer.set_init_simplex(initialGuess);

  // Run optimization
  Vector optimizedParameters = optimizer.optimize();
  std::cout << "Optimized parameters: k = " << optimizedParameters(0)
            << ", delta = " << optimizedParameters(1)
            << ", b = " << optimizedParameters(2) << std::endl;

  return 0;
}
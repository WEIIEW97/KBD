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
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>

#include "../src/optimizer.h"

using namespace std;
using namespace Eigen;

// // Templated Objective Function Interface
// template <typename Scalar>
// class ObjectiveFunction {
// public:
//   virtual Scalar evaluate(const Matrix<Scalar, Dynamic, 1>& x) const = 0;
// };

// // Rosenbrock's Banana Function as an example of an ObjectiveFunction
// template <typename Scalar>
// class RosenbrockFunction : public ObjectiveFunction<Scalar> {
// public:
//   Scalar evaluate(const Matrix<Scalar, Dynamic, 1>& x) const override {
//     Scalar a = 1.0, b = 100.0;
//     return pow(a - x(0), 2) + b * pow(x(1) - x(0) * x(0), 2);
//   }
// };

// // Templated Nelder-Mead Optimizer Class
// template <typename Scalar>
// class NelderMeadOptimizer {
// private:
//   vector<Matrix<Scalar, Dynamic, 1>> simplex;
//   Scalar alpha, gamma, rho, sigma;
//   int maxIter;

// public:
//   NelderMeadOptimizer(Scalar alpha = 1.0, Scalar gamma = 2.0, Scalar rho = 0.5,
//                       Scalar sigma = 0.5, int maxIter = 100)
//       : alpha(alpha), gamma(gamma), rho(rho), sigma(sigma), maxIter(maxIter) {}

//   void
//   setInitialSimplex(const vector<Matrix<Scalar, Dynamic, 1>>& initialSimplex) {
//     simplex = initialSimplex;
//   }

//   Matrix<Scalar, Dynamic, 1> optimize(const ObjectiveFunction<Scalar>& func) {
//     int n = simplex[0].size();

//     for (int iter = 0; iter < maxIter; ++iter) {
//       // Sort simplex by objective value
//       sort(simplex.begin(), simplex.end(),
//            [&](const Matrix<Scalar, Dynamic, 1>& a,
//                const Matrix<Scalar, Dynamic, 1>& b) {
//              return func.evaluate(a) < func.evaluate(b);
//            });

//       // Calculate centroid of all but worst point
//       Matrix<Scalar, Dynamic, 1> centroid = Matrix<Scalar, Dynamic, 1>::Zero(n);
//       for (int i = 0; i < simplex.size() - 1; ++i) {
//         centroid += simplex[i];
//       }
//       centroid /= (simplex.size() - 1);

//       // Reflection
//       Matrix<Scalar, Dynamic, 1> worst = simplex.back();
//       Matrix<Scalar, Dynamic, 1> reflected =
//           centroid + alpha * (centroid - worst);
//       Scalar reflected_val = func.evaluate(reflected);

//       if (reflected_val < func.evaluate(simplex[0])) {
//         // Expansion
//         Matrix<Scalar, Dynamic, 1> expanded =
//             centroid + gamma * (reflected - centroid);
//         if (func.evaluate(expanded) < reflected_val) {
//           simplex.back() = expanded;
//         } else {
//           simplex.back() = reflected;
//         }
//       } else if (reflected_val < func.evaluate(worst)) {
//         // Accept reflection
//         simplex.back() = reflected;
//       } else {
//         // Contraction
//         Matrix<Scalar, Dynamic, 1> contracted =
//             centroid + rho * (worst - centroid);
//         if (func.evaluate(contracted) < func.evaluate(worst)) {
//           simplex.back() = contracted;
//         } else {
//           // Shrink
//           for (int i = 1; i < simplex.size(); ++i) {
//             simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0]);
//           }
//         }
//       }
//     }

//     return simplex[0]; // Return the best point
//   }
// };

// template <typename T>
// T rosenbrock(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) {
//   T a = 1.0, b = 100.0;
//   return (a - x(0)) * (a - x(0)) +
//          b * (x(1) - x(0) * x(0)) * (x(1) - x(0) * x(0));
// };

// int main() {
//   //   // Create optimizer
//   //   NelderMeadOptimizer<double> optimizer;

//   //   // Define the initial simplex
//   //   vector<VectorXd> initialSimplex = {
//   //       MatrixXd::Constant(2, 1, -1.2),
//   //       MatrixXd::Constant(2, 1, 0.0),
//   //       MatrixXd::Constant(2, 1, 2.0)
//   //   };
//   //   optimizer.setInitialSimplex(initialSimplex);

//   //   // Define the objective function
//   //   RosenbrockFunction<double> func;

//   //   // Perform optimization
//   //   MatrixXd result = optimizer.optimize(func);
//   //   cout << "Optimized Result: (" << result(0) << ", " << result(1) << ")" <<
//   //   endl;
//   using T = double; // Define the type of scalars used
//   Eigen::Matrix<T, Eigen::Dynamic, 1> initial_guess(
//       2);                     // Rosenbrock function is 2D
//   initial_guess << -1.2, 1.0; // Set initial guess

//   // Create an instance of the optimizer
//   kbd::NelderMeadOptimizer<T> optimizer(
//       rosenbrock<T>, // Pass the Rosenbrock function as the objective function
//       initial_guess, // Initial guess
//       1.0,           // alpha
//       2.0,           // gamma
//       0.5,           // rho
//       0.5            // sigma
//   );

//   // Optimize and retrieve the result
//   Eigen::Matrix<T, Eigen::Dynamic, 1> result = optimizer.optimize(10000, 1e-6);

//   // Print the result
//   std::cout << "Optimized result: (" << result(0) << ", " << result(1) << ")"
//             << std::endl;
//   std::cout << "Function value at minimum: " << rosenbrock<T>(result)
//             << std::endl;

//   return 0;
// }


template<typename Scalar>
class NelderMeadOptimizer {
private:
    vector<Matrix<Scalar, Dynamic, 1>> simplex;
    Scalar alpha, gamma, rho, sigma;
    int maxIter;
    std::function<Scalar(const Matrix<Scalar, Dynamic, 1>&)> objective_func;

public:
    NelderMeadOptimizer(std::function<Scalar(const Matrix<Scalar, Dynamic, 1>&)> func,
                        Scalar alpha = 1.0, Scalar gamma = 2.0, Scalar rho = 0.5, Scalar sigma = 0.5, int maxIter = 10000)
        : objective_func(func), alpha(alpha), gamma(gamma), rho(rho), sigma(sigma), maxIter(maxIter) {}

    void setInitialSimplex(const vector<Matrix<Scalar, Dynamic, 1>>& initialSimplex) {
        simplex = initialSimplex;
    }

    Matrix<Scalar, Dynamic, 1> optimize() {
        int n = simplex[0].size();

        for (int iter = 0; iter < maxIter; ++iter) {
            // Sort simplex by objective value
            sort(simplex.begin(), simplex.end(), [&](const Matrix<Scalar, Dynamic, 1>& a, const Matrix<Scalar, Dynamic, 1>& b) {
                return objective_func(a) < objective_func(b);
            });

            // Calculate centroid of all but worst point
            Matrix<Scalar, Dynamic, 1> centroid = Matrix<Scalar, Dynamic, 1>::Zero(n);
            for (int i = 0; i < simplex.size() - 1; ++i) {
                centroid += simplex[i];
            }
            centroid /= (simplex.size() - 1);

            // Reflection
            Matrix<Scalar, Dynamic, 1> worst = simplex.back();
            Matrix<Scalar, Dynamic, 1> reflected = centroid + alpha * (centroid - worst);
            Scalar reflected_val = objective_func(reflected);

            if (reflected_val < objective_func(simplex[0])) {
                // Expansion
                Matrix<Scalar, Dynamic, 1> expanded = centroid + gamma * (reflected - centroid);
                if (objective_func(expanded) < reflected_val) {
                    simplex.back() = expanded;
                } else {
                    simplex.back() = reflected;
                }
            } else if (reflected_val < objective_func(worst)) {
                // Accept reflection
                simplex.back() = reflected;
            } else {
                // Contraction
                Matrix<Scalar, Dynamic, 1> contracted = centroid + rho * (worst - centroid);
                if (objective_func(contracted) < objective_func(worst)) {
                    simplex.back() = contracted;
                } else {
                    // Shrink
                    for (int i = 1; i < simplex.size(); ++i) {
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0]);
                    }
                }
            }
        }

        return simplex[0]; // Return the best point
    }
};

// Example usage with a Rosenbrock function defined as a lambda
int main() {
    auto rosenbrock = [](const Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
        return pow(1 - x[0], 2) + 100 * pow(x[1] - x[0]*x[0], 2);
    };

    NelderMeadOptimizer<double> optimizer(rosenbrock);
    vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> initialSimplex = {
        Eigen::Vector2d(-1.2, 1),
        Eigen::Vector2d(0.0, 0.0),
        Eigen::Vector2d(2.0, 2.0)
    };
    optimizer.setInitialSimplex(initialSimplex);
    Eigen::Vector2d result = optimizer.optimize();
    cout << "Optimized result: (" << result[0] << ", " << result[1] << ")" << endl;
}
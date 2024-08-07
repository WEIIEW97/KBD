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

#include <Eigen/Dense>
#include <fmt/ostream.h>

int main() {
    Eigen::Matrix<double, 2, 2> matrix;
    matrix << 1, 2, 3, 4;
    fmt::print("{:=>50}\n", "");
    fmt::print("Matrix:\n{}\n", matrix);

    Eigen::Vector3d vector;
    vector << 1, 2, 3;
    fmt::print("{:=>50}\n", "");
    fmt::print("Vector:\n{}\n", vector);

    std::vector<int> vec = {1, 2, 3, 4, 5};
    fmt::print("{:=>50}\n", "");
    fmt::print("Vector: {}\n", fmt::join(vec, ", "));
    fmt::print("{:=>50}\n", "");

    return 0;
}
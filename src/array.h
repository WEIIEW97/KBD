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

#ifndef KBD_ARRAY_H
#define KBD_ARRAY_H

#include <Eigen/Dense>
#include <arrow/api.h>
#include <fmt/core.h>
#include <fmt/format.h>

namespace kbd {
  // Template alias for Eigen Matrix
  template <typename T>
  using ndArray = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template <typename T>
  Eigen::Array<T, Eigen::Dynamic, 1> ArrowDoubleArrayToEigen(
      const std::shared_ptr<arrow::DoubleArray>& double_array) {
    // Create an Eigen array of appropriate size
    Eigen::Array<T, Eigen::Dynamic, 1> eigen_array(double_array->length());

    // Copy data from Arrow array to Eigen array
    for (auto i = 0; i < double_array->length(); ++i) {
      eigen_array(i) = static_cast<T>(double_array->Value(i));
    }

    return eigen_array;
  }

  template <typename T>
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> ArrowDoubleArrayToEigenMap(
      const std::shared_ptr<arrow::DoubleArray>& double_array) {

    // Get the raw data pointer
    const T* data_ptr = double_array->raw_values();

    // Create an Eigen::Map object to wrap the raw data pointer
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> eigen_array(
        data_ptr, double_array->length());

    return eigen_array;
  }

} // namespace kbd

#endif // KBD_ARRAY_H

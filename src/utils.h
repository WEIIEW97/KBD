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

#ifndef KBD_UTILS_H
#define KBD_UTILS_H

#include "array.h"
#include "config.h"

#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <iostream>
namespace kbd {
  namespace fs = std::filesystem;
  using json = nlohmann::json;

  template <typename T>
  ndArray<T> load_raw(const std::string& path, int h, int w) {
    // Open the file in binary mode
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "Cannot open file: " << path << std::endl;
      return ndArray<T>();
    }

    // Create a buffer to hold the raw data
    std::vector<T> buffer(h * w);

    // Read the data from the file
    file.read(reinterpret_cast<char*>(buffer.data()), h * w * sizeof(T));
    if (!file) {
      std::cerr << "Error reading file: " << path << std::endl;
      return ndArray<T>();
    }
    file.close();

    Eigen::Map<ndArray<T>> matrix(buffer.data(), h, w);

    // Copy the data to an Eigen::Matrix (optional, to ensure the data is owned
    // by the matrix)
    ndArray<T> eigen_matrix = matrix;

    return eigen_matrix;
  }

  template <typename T>
  void save_raw(const std::string& path, const ndArray<T>& matrix) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "Cannot open file for writing: " << path << std::endl;
    }

    file.write(reinterpret_cast<const char*>(matrix.data()),
               matrix.size() * sizeof(T));
    if (!file) {
      std::cerr << "Error writing to file: " << path << std::endl;
    }
    file.close();
  }

  template <typename T, int Rows, int Cols>
  double calculate_median(const Eigen::Matrix<T, Rows, Cols>& matrix) {
    std::vector<T> data(matrix.data(), matrix.data() + matrix.size());
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return static_cast<double>(data[data.size() / 2]);
  }

  template <typename T, int Rows, int Cols>
  double calculate_median(const Eigen::Block<T, Rows, Cols>& matrix) {
    std::vector<T> data(matrix.data(), matrix.data() + matrix.size());
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return static_cast<double>(data[data.size() / 2]);
  }

  template <typename T, int Rows, int Cols>
  double calculate_mean(const Eigen::Matrix<T, Rows, Cols>& matrix) {
    return matrix.template cast<double>().mean();
  }

  template <typename T, int Rows, int Cols>
  double calculate_mean(const Eigen::Block<T, Rows, Cols>& matrix) {
    /// this would make zero value pixel into calculation
    return matrix.template cast<double>().mean();
  }

  template <typename T, int Rows, int Cols>
  double calculate_mean_nonzero(const Eigen::Block<T, Rows, Cols>& matrix) {
    // Use the derived matrix as an Array for element-wise operations
    Eigen::MatrixXd m;
    m = matrix.template cast<double>();
    auto mask = m.array() != 0;
    double sum = m.sum();

    int count = static_cast<int>(mask.count());

    return count > 0 ? sum / count : 0; // Prevent division by zero
  }

  template <typename Derived>
  Derived weighted_mse(
      const Eigen::Array<Derived, Eigen::Dynamic, Eigen::Dynamic>& Y,
      const Eigen::Array<Derived, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Array<Derived, Eigen::Dynamic, Eigen::Dynamic>& weights) {
    auto squared_diff = (Y - X).square();
    double total_weights_squared_sum = (squared_diff * weights).sum();
    double total_weights = weights.sum();
    return total_weights_squared_sum / total_weights;
  }

  std::vector<std::string> retrieve_folder_names(const std::string& path);
  std::vector<std::string> retrieve_file_names(const std::string& path);
  void copy_files_in_directory(const std::string& src, const std::string& dst);
  void
  save_arrays_to_json(const std::string& path,
                      const Eigen::Array<uint16_t, Eigen::Dynamic, 1>& arr1d,
                      const Eigen::MatrixXd& arr2d);
  void save_arrays_to_json_debug(
      const std::string& path,
      const Eigen::Array<uint16_t, Eigen::Dynamic, 1>& arr1d,
      const Eigen::MatrixXd& arr2d, int range_start, double compensate_dist);
  std::map<std::string, double>
  calculate_mean_value(const std::string& file_path,
                       const std::vector<std::string>& folders,
                       const Config& default_configs);

} // namespace kbd

#endif // KBD_UTILS_H

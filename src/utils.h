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

#include "shared.h"
#include "array.h"
#include "config.h"

#include <nlohmann/json.hpp>

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

    // Calculate the total number of elements
    int total_elements = h * w;

    // Create a buffer to hold the raw data
    std::vector<T> buffer(total_elements);

    // Read the data from the file
    file.read(reinterpret_cast<char*>(buffer.data()),
              total_elements * sizeof(T));
    if (!file) {
      std::cerr << "Error reading file: " << path << std::endl;
      return ndArray<T>();
    }

    // Close the file
    file.close();

    // Map the buffer to an Eigen matrix
    ndArray<T> matrix(h, w);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        matrix(i, j) = buffer[i * w + j];
      }
    }

    return matrix;
  }

  std::vector<std::string> retrieve_folder_names(const std::string& path);
  std::vector<std::string> retrieve_file_names(const std::string& path);
  void copy_files_in_directory(const std::string& src, const std::string& dst);
  void save_arrays_to_json(const std::string& path,
                           const std::vector<uint16_t>& arr1d,
                           const Eigen::MatrixXd& arr2d);
  std::map<std::string, double> calculate_mean_value(
      const std::string& root_path, const std::vector<std::string>& folders,
      const Config& default_configs, bool is_median);
} // namespace kbd

#endif // KBD_UTILS_H

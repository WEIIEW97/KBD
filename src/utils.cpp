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
#include "utils.h"
#include <filesystem>

namespace kbd {
  std::vector<std::string> retrieve_folder_names(const std::string& path) {
    std::vector<std::string> folder_names;

    // Iterate through the directory entries in the given path
    for (const auto& entry : fs::directory_iterator(path)) {
      // Check if the entry is a directory
      if (fs::is_directory(entry.status())) {
        folder_names.push_back(entry.path().filename().string());
      }
    }

    return folder_names;
  }

  std::vector<std::string> retrieve_file_names(const std::string& path) {
    std::vector<std::string> file_names;
    for (const auto& entry : fs::directory_iterator(path)) {
      if (fs::is_regular_file(entry.status())) {
        file_names.push_back((entry.path().filename().string()));
      }
    }

    return file_names;
  }

  void copy_files_in_directory(const std::string& src, const std::string& dst) {
    // Create the destination directory if it does not exist
    fs::create_directories(dst);

    // Retrieve the list of files in the source directory
    auto files = retrieve_file_names(src);

    // Copy each file from src to dst
    for (const auto& file : files) {
      fs::path source = fs::path(src) / file;
      fs::path destination = fs::path(dst) / file;
      fs::copy_file(
          source, destination,
          fs::copy_options::overwrite_existing);
    }
  }

  void save_arrays_to_json(const std::string& path,
                           const std::vector<uint16_t>& arr1d,
                           const Eigen::MatrixXd& arr2d) {
    json J;
    for (auto i : arr1d) {
      J["disp_nodes"].push_back(i);
    }
    for (int i = 0; i < arr2d.rows(); ++i) {
      json row = json::array();
      for (int j = 0; j < arr2d.cols(); ++j) {
        row.push_back(arr2d(i, j));
      }
      J["kbd_params"].push_back(row);
    }
    std::ofstream file(path);
    file << J.dump(4);
    file.close();
  }

  std::map<std::string, double> calculate_mean_value(
      const std::string& root_path, const std::vector<std::string>& folders,
      const Config& default_configs, bool is_median) {
    std::map<std::string, double> dist_dict;
    auto subfix = default_configs.SUBFIX;
    auto anchor_point = default_configs.ANCHOR_POINT;
    auto h = default_configs.H;
    auto w = default_configs.W;
    
    for (const auto& folder : folders) {
      std::string distance = folder.substr(0, folder.find("_"));
      fs::path raw_path = fs::path(root_path) / folder / subfix;
      std::vector<double> mean_dist_holder;

      for (const auto& entry : fs::directory_iterator(raw_path)) {
        if (entry.is_regular_file()) {
          auto raw = load_raw<uint16_t>(entry.path().string(), h, w);
          auto valid_raw = raw.block<uint16_t>(anchor_point[0] - 25,
                                               anchor_point[1] - 25, h, w);
          double mu;
          if (is_median) {
            std::vector<double> temp(valid_raw.data(),
                                     valid_raw.data() + valid_raw.size());
            std::nth_element(temp.begin(), temp.begin() + temp.size() / 2,
                             temp.end());
            mu = temp[temp.size() / 2];
          } else {
            mu = valid_raw.mean();
          }
          mean_dist_holder.push_back(mu);
        }
      }
      double final_mu = std::accumulate(mean_dist_holder.begin(),
                                        mean_dist_holder.end(), 0.0) /
                        mean_dist_holder.size();
      dist_dict[distance] = final_mu;
    }
    return dist_dict;
  }
} // namespace kbd

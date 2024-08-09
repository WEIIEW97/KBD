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

#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
#include "../src/workflow.h"
#include "../src/utils.h"
#include "../src/ops/modifier.h"
#include <iostream>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/color.h>

namespace fs = std::filesystem;

int main() {
  // auto cwd = fs::current_path(); // note that this is the binary path
  std::string home_path =
      "/home/william/extdisk/data/KBD_analysis/uphill curve";
  const std::string root_path = home_path + "/N9LAZG24GN0007";
  const std::string csv_path =
      home_path + "/N9LAZG24GN0007/depthquality_2024-07-24.csv";
  const std::string file_path = home_path + "/N9LAZG24GN0007/image_data";
  const std::string transformed_file_path =
      home_path + "/N9LAZG24GN0007/image_data_lc++";
  if (!fs::exists(transformed_file_path)) {
    // Create the directory since it does not exist
    if (fs::create_directory(transformed_file_path)) {
      std::cout << "Directory created successfully: " << transformed_file_path
                << std::endl;
    } else {
      std::cout << "Failed to create directory." << std::endl;
    }
  } else {
    std::cout << "Directory already exists: " << transformed_file_path
              << std::endl;
  }
  kbd::Config default_configs = kbd::Config();
  kbd::JointSmoothArguments args = kbd::JointSmoothArguments();

  kbd::LinearWorkflow workflow;
  std::array<int, 2> search_range = {600, 1100};
  std::array<double, 2> cd_range = {100, 400};

  workflow.preprocessing(file_path, csv_path, default_configs, args);
  bool export_original = false;

  if (!workflow.first_check() || !workflow.pass_or_not()) {
    auto [eval_res, acceptance] = workflow.eval();
    std::cout << "acceptance rate: " << acceptance << std::endl;
    if (acceptance < default_configs.EVAL_WARNING_RATE) {
      fmt::print(fmt::fg(fmt::color::red),
                 "*********** WARNING *************\n");
      fmt::print(fmt::fg(fmt::color::red),
                 "Please be really cautious since the acceptance rate is {},\n",
                 acceptance);
      fmt::print(fmt::fg(fmt::color::red),
                 "This may not be the ideal data to be tackled with.\n");
      fmt::print(fmt::fg(fmt::color::red),
                 "*********** END OF WARNING *************\n");
    }
    kbd::LinearWorkflow::grid_search GridSearcher(&workflow);
    GridSearcher.optimize_params(search_range, cd_range);
    auto [matrix, rng_start, cd] = GridSearcher.get_results();
    
  }
  return 0;
}

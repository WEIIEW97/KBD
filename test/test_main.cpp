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
#include <Eigen/Dense>
#include <Eigen/Core>

namespace fs = std::filesystem;

#define rng_end 3000

enum class ReturnStatus : int {
  NO_NEED_KBD,
  KBD_AND_PASSED,
  KBD_BUT_FAILED,
  KBD_BUT_ORIGINAL_BETTER,
  ERROR,
};

int main() {
  // auto cwd = fs::current_path(); // note that this is the binary path
  std::string home_path = "D:/william/data/KBD";
  const std::string base_path = home_path + "/Z06FLAZG24GN0347";
  const std::string csv_path =
      home_path + "/Z06FLAZG24GN0347/depthquality_2024-09-30.csv";
  const std::string file_path = home_path + "/Z06FLAZG24GN0347/image_data";
  const std::string transformed_file_path =
      home_path + "/Z06FLAZG24GN0347/image_data_lc++";
  const std::string mode = "N9";
  int cy = 205, cx = 310;
  float focal_scalar = 1.0f;
  bool apply_global = false;
  if (file_path.empty() || csv_path.empty() || transformed_file_path.empty()) {
    fmt::print("You have to make sure arguments are not empty!/n");
  }

  if (!fs::exists(transformed_file_path)) {
    if (fs::create_directory(transformed_file_path)) {
      fmt::print("Directory created successfully: {}/n", transformed_file_path);
    } else {
      fmt::print("Failed to create directory. /n");
    }
  } else {
    fmt::print("Directory already exists: {}/n", transformed_file_path);
  }
  kbd::Config default_configs = kbd::Config(mode);
  kbd::JointSmoothArguments args = kbd::JointSmoothArguments(mode);

  if (cy != 0 && cx != 0)
    default_configs.ANCHOR_POINT = {cy, cx};

  if (focal_scalar != 0)
    default_configs.FOCAL_MULTIPLIER = focal_scalar;

  kbd::LinearWorkflow workflow(default_configs, args);
  std::array<int, 2> search_range = {600, 1100};
  std::array<double, 2> cd_range = {100, 400};

  workflow.preprocessing(file_path, csv_path);
  bool export_original = false;
  auto global_judge = (apply_global ? "global" : "local");
  auto output_json_name = fmt::format(
      "{}_{}.json", default_configs.BASE_OUTPUT_JSON_FILE_NAME_PREFIX,
      global_judge);
  const std::string dumped_json_path = base_path + "/" + output_json_name;
  Eigen::Matrix<double, 5, 5> m;

  ReturnStatus status = ReturnStatus::ERROR;

  if (!workflow.first_check() || !workflow.pass_or_not()) {
    auto [eval_res, acceptance] = workflow.eval();
    std::cout << "acceptance rate: " << acceptance << std::endl;
    if (acceptance < default_configs.EVAL_WARNING_RATE) {
      fmt::print(fmt::fg(fmt::color::red),
                 "*********** WARNING *************/n");
      fmt::print(fmt::fg(fmt::color::red),
                 "Please be really cautious since the acceptance rate is {},/n",
                 acceptance);
      fmt::print(fmt::fg(fmt::color::red),
                 "This may not be the ideal data to be tackled with./n");
      fmt::print(fmt::fg(fmt::color::red),
                 "*********** END OF WARNING *************/n");
    }
    kbd::LinearWorkflow::grid_search GridSearcher(&workflow);
    GridSearcher.optimize_params(search_range, cd_range);
    auto [matrix, rng_start, cd] = GridSearcher.get_results();
    std::array<int, 2> best_range = {rng_start, rng_end};
    args.compensate_dist = static_cast<int>(std::round(cd));
    args.disjoint_depth_range = best_range;
    bool apply_kbd = workflow.final_check(matrix, best_range, cd);
    if (apply_kbd) {
      auto [disp_nodes, reversed_matrix] =
          workflow.pivot(matrix, best_range, cd);
      kbd::save_arrays_to_json(dumped_json_path, disp_nodes, reversed_matrix);
      // kbd::save_arrays_to_json_debug(dumped_json_path, disp_nodes,
      //                                reversed_matrix, rng_start, cd);
      m = matrix;

      fmt::print("Working done for the optimization part!/n");
      if (workflow.final_pass_) {
        status = ReturnStatus::KBD_AND_PASSED;
      } else {
        status = ReturnStatus::KBD_BUT_FAILED;
      }
    } else {
      auto [disp_nodes, reversed_matrix] = workflow.export_default();
      kbd::save_arrays_to_json(dumped_json_path, disp_nodes, reversed_matrix);
      // kbd::save_arrays_to_json_debug(dumped_json_path, disp_nodes,
      //                                reversed_matrix, rng_start, cd);
      fmt::print("Working done for the optimization part!/n");
      export_original = true;
      m = matrix;
      status = ReturnStatus::KBD_BUT_ORIGINAL_BETTER;
    }
  } else {
    auto [disp_nodes, reversed_matrix] = workflow.export_default();
    kbd::save_arrays_to_json(dumped_json_path, disp_nodes, reversed_matrix);
    fmt::print("Working done for the optimization part!/n");
    export_original = true;
    m = workflow.full_kbd_params5x5_;
    status = ReturnStatus::NO_NEED_KBD;
  }

  fmt::print("Begin copying ... /n");
  kbd::ops::parallel_copy(file_path, transformed_file_path, default_configs);

  fmt::print("Begin transformation ... /n");
  if (!export_original) {
    kbd::ops::parallel_transform(
        transformed_file_path, m, workflow.get_focal_val(),
        workflow.get_baseline_val(), default_configs, args);
  }
  fmt::print("All tasks done! ... /n");

  return 0;
}

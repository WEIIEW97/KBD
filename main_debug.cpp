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
#include "src/workflow.h"
#include "src/utils.h"
#include "src/ops/modifier.h"
#include <iostream>
#include <fmt/core.h>
#include <fmt/color.h>
#include <filesystem>
#include <boost/program_options.hpp>

namespace fs = std::filesystem;
namespace po = boost::program_options;

#define rng_end 3000

enum class ReturnStatus : int {
  NO_NEED_KBD,
  KBD_AND_PASSED,
  KBD_BUT_FAILED,
  KBD_BUT_ORIGINAL_BETTER,
  ERROR,
};

ReturnStatus pshyco(const std::string& file_path, const std::string& csv_name,
                    const std::string& mode, int cy, int cx, bool apply_global = false) {
  fs::path fs_file_path(file_path);
  auto base_path = fs_file_path.parent_path();
  auto csv_path = base_path / csv_name;
  auto transformed_file_path = base_path / "transformed_data_l";

  if (file_path.empty() || csv_path.empty() || transformed_file_path.empty()) {
    fmt::print("You have to make sure arguments are not empty!\n");
    return ReturnStatus::ERROR;
  }

  if (!fs::exists(transformed_file_path)) {
    if (fs::create_directory(transformed_file_path)) {
      fmt::print("Directory created successfully: {}\n",
                 transformed_file_path.string());
    } else {
      fmt::print("Failed to create directory. \n");
    }
  } else {
    fmt::print("Directory already exists: {}\n",
               transformed_file_path.string());
  }
  kbd::Config default_configs = kbd::Config(mode);
  kbd::JointSmoothArguments args = kbd::JointSmoothArguments(mode);

  if (cy != 0 && cx != 0)
    default_configs.ANCHOR_POINT = {cy, cx};

  kbd::LinearWorkflow workflow(default_configs, args);
  std::array<int, 2> search_range = {600, 1100};
  std::array<double, 2> cd_range = {100, 400};

  workflow.preprocessing(file_path, csv_path.string());
  bool export_original = false;
  auto global_judge = (apply_global ? "global" : "local");
  auto output_json_name = fmt::format(
      "{}_{}.json", default_configs.BASE_OUTPUT_JSON_FILE_NAME_PREFIX,
      global_judge);
  const std::string dumped_json_path =
      base_path.string() + "/" + output_json_name;
  Eigen::Matrix<double, 5, 5> m;

  ReturnStatus status = ReturnStatus::ERROR;

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
    std::array<int, 2> best_range = {rng_start, rng_end};
    args.compensate_dist = static_cast<int>(std::round(cd));
    args.disjoint_depth_range = best_range;
    bool apply_kbd = workflow.final_check(matrix, best_range, cd);
    if (apply_kbd) {
      auto [disp_nodes, reversed_matrix] =
          workflow.pivot(matrix, best_range, cd);
      // kbd::save_arrays_to_json(dumped_json_path, disp_nodes, reversed_matrix);
      kbd::save_arrays_to_json_debug(dumped_json_path, disp_nodes,
                                     reversed_matrix, rng_start, cd);
      m = matrix;

      fmt::print("Working done for the optimization part!\n");
      if (workflow.final_pass_) {
        status = ReturnStatus::KBD_AND_PASSED;
      } else {
        status = ReturnStatus::KBD_BUT_FAILED;
      }
    } else {
      auto [disp_nodes, reversed_matrix] = workflow.export_default();
      // kbd::save_arrays_to_json(dumped_json_path, disp_nodes, reversed_matrix);
      kbd::save_arrays_to_json_debug(dumped_json_path, disp_nodes,
                                     reversed_matrix, rng_start, cd);
      fmt::print("Working done for the optimization part!\n");
      export_original = true;
      m = matrix;
      status = ReturnStatus::KBD_BUT_ORIGINAL_BETTER;
    }
  } else {
    auto [disp_nodes, reversed_matrix] = workflow.export_default();
    kbd::save_arrays_to_json(dumped_json_path, disp_nodes, reversed_matrix);
    fmt::print("Working done for the optimization part!\n");
    export_original = true;
    m = workflow.full_kbd_params5x5_;
    status = ReturnStatus::NO_NEED_KBD;
  }

  fmt::print("Begin copying ... \n");
  kbd::ops::parallel_copy(file_path, transformed_file_path.string(),
                          default_configs);

  fmt::print("Begin transformation ... \n");
  if (!export_original) {
    kbd::ops::parallel_transform(
        transformed_file_path.string(), m, workflow.get_focal_val(),
        workflow.get_baseline_val(), default_configs, args);
  }
  fmt::print("All tasks done! ... \n");

  return status;
}

int main(int argc, char** argv) {

  std::string file_path, csv_name, mode;
  bool apply_global = false;
  int cy, cx;

  //========= Handling Program options =========
  po::options_description desc("Allowed options");
  desc.add_options()("file_path,f",
                     po::value<std::string>(&file_path)->default_value(""),
                     "root directory for the raw data, e.g. 'image_data/'")(
      "csv_name,c", po::value<std::string>(&csv_name)->default_value(""),
      "path to the .csv measured information.")(
      "mode,m", po::value<std::string>(&mode)->default_value("N9"),
      "KBD program target mode.")(
      "anchor_y,y", po::value<int>(&cy)->default_value(0), "anchor point y")(
      "anchor_x,x", po::value<int>(&cx)->default_value(0),
      "anchor point x")("apply_global,g", po::bool_switch(&apply_global),
                        "apply optimization globally");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  auto return_status = pshyco(file_path, csv_name, mode, cy, cx);

  if (return_status == ReturnStatus::ERROR) {
    fs::path fs_file_path(file_path);
    auto base_path = fs_file_path.parent_path();
    auto error_path = base_path / "error.log";

    std::ofstream out(error_path);
    if (!out.is_open()) {
      std::cerr << "failed to open file." << std::endl;
      return 1;
    }
    out << "Configuration is wrong! Application cannot run!" << std::endl;
    out.close();
  } else if (return_status == ReturnStatus::KBD_BUT_FAILED) {
    fs::path fs_file_path(file_path);
    auto base_path = fs_file_path.parent_path();
    auto failed_path = base_path / "failed.log";

    std::ofstream out(failed_path);
    if (!out.is_open()) {
      std::cerr << "failed to open file." << std::endl;
      return 1;
    }
    out << "This device cannot pass the metric even after kbd. Please recheck "
           "the device itself or your operation."
        << std::endl;
    out.close();
  }

  return 0;
}

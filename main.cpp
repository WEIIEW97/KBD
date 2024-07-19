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

int main(int argc, char** argv) {

  std::string file_path, csv_path, transformed_file_path;
  bool apply_global = true;

  //========= Handling Program options =========
  po::options_description desc("Allowed options");
  desc.add_options()("file_path,f",
                     po::value<std::string>(&file_path)->default_value(""),
                     "root directory for the raw data, e.g. 'image_data/'")(
      "csv_path,c", po::value<std::string>(&csv_path)->default_value(""),
      "path to the .csv measured information.")(
      "output_path,t",
      po::value<std::string>(&transformed_file_path)->default_value(""),
      "output directory for the transformed data.")(
      "apply_global,g", po::bool_switch(&apply_global),
      "apply optimization globally");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (file_path.empty() || csv_path.empty() || transformed_file_path.empty()) {
    fmt::print("You have to make sure arguments are not empty!\n");
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  auto home_path = fs::path(file_path).parent_path().string();

  if (!fs::exists(transformed_file_path)) {
    // Create the directory since it does not exist
    if (fs::create_directory(transformed_file_path)) {
      fmt::print("Directory created successfully: {}\n", transformed_file_path);
    } else {
      fmt::print("Failed to create directory. \n");
    }
  } else {
    fmt::print("Directory already exists: {}\n", transformed_file_path);
  }
  kbd::Config default_configs = kbd::Config();
  kbd::JointSmoothArguments args = kbd::JointSmoothArguments();

  args.apply_global = apply_global;

  kbd::LinearWorkflow workflow;

  workflow.preprocessing(file_path, csv_path, default_configs, args);
  auto [eval_res, acceptance] = workflow.eval(default_configs);

  if (acceptance < default_configs.EVAL_WARNING_RATE) {
    fmt::print(fmt::fg(fmt::color::red), "*********** WARNING *************\n");
    fmt::print(fmt::fg(fmt::color::red),
               "Please be really cautious since the acceptance rate is {},\n",
               acceptance);
    fmt::print(fmt::fg(fmt::color::red),
               "This may not be the ideal data to be tackled with.\n");
    fmt::print(fmt::fg(fmt::color::red),
               "*********** END OF WARNING *************\n");
  }

  workflow.optimize();
  workflow.extend_matrix();
  auto [disp_nodes, param_matrix] = workflow.pivot();

  auto global_judge = (apply_global ? "global" : "local");
  auto output_json_name = fmt::format(
      "{}_{}.json", default_configs.BASE_OUTPUT_JSON_FILE_NAME_PREFIX,
      global_judge);
  const std::string dumped_json_path = home_path + "/" + output_json_name;
  kbd::save_arrays_to_json(dumped_json_path, disp_nodes, param_matrix);

  fmt::print("Working done for the optimization part!\n");

  fmt::print("Begin copying ... \n");
  kbd::ops::parallel_copy(file_path, transformed_file_path, default_configs);

  fmt::print("Begin transformation ... \n");
  kbd::ops::parallel_transform(
      transformed_file_path, param_matrix, workflow.get_focal_val(),
      workflow.get_baseline_val(), default_configs, args);

  fmt::print("All tasks done! ... \n");
  return 0;
}

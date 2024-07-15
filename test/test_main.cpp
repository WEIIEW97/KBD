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
#include "../src/table.h"
#include "../src/optimizer.h"
#include "../src/format.h"
#include "../src/utils.h"
#include <iostream>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
  const std::string root_path = "/home/william/Codes/KBD/data/N09ASH24DH0050";
  const std::string csv_path = "/home/william/Codes/KBD/data/N09ASH24DH0050/depthquality_2024-07-09.csv";
  const std::string file_path = "/home/william/Codes/KBD/data/N09ASH24DH0050/image_data";
  kbd::Config default_configs = kbd::Config();

  auto table_parser = kbd::ArrowTableReader();
  auto df = table_parser.read_csv(csv_path);
  auto trimmed_df = table_parser.trim_table(default_configs.MAPPED_PAIR_DICT);
  auto dist_dict = kbd::calculate_mean_value(file_path, kbd::retrieve_folder_names(file_path), default_configs);
  auto status = table_parser.map_table(trimmed_df, default_configs, dist_dict);

  auto col_names = trimmed_df->ColumnNames();
  for (const auto& v : col_names) {
    std::cout << v << "\n";
  }

  auto gt_arrow_col = trimmed_df->GetColumnByName(default_configs.GT_DIST_NAME);
  auto est_arrow_col = trimmed_df->GetColumnByName(default_configs.AVG_DIST_NAME);
  auto gt_double = std::static_pointer_cast<arrow::DoubleArray>(gt_arrow_col->chunk(0));
  auto est_double = std::static_pointer_cast<arrow::DoubleArray>(est_arrow_col->chunk(0));
  Eigen::Map<const Eigen::ArrayXd> gt_eigen_array(gt_double->raw_values(), gt_double->length());
  Eigen::Map<const Eigen::ArrayXd> est_eigen_array(est_double->raw_values(), est_double->length());

  std::array<int, 2> disjoint_depth_range = {600, 3000};
  auto linear_kbd_optim = kbd::JointLinearSmoothingOptimizer(
    gt_eigen_array.cast<double>(),
    est_eigen_array.cast<double>(),
    table_parser.focal_,
    table_parser.baseline_,
    disjoint_depth_range
  );

  auto [lm1, kbd_res, km2] = linear_kbd_optim.run();
  
  return 0;
}

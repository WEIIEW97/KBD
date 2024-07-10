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

#include "factory.h"
#include "config.h"

namespace kbd {
  std::tuple<std::shared_ptr<arrow::Table>, double, double>
  preprocessing(const std::string& path, const std::string& table_path,
                const Config& default_configs, bool is_median) {
    auto all_distannces = retrieve_folder_names(path);
    auto mean_dists =
        calculate_mean_value(path, all_distannces, default_configs, is_median);

    ArrowTableReader arrow_table_reader;
    auto df = arrow_table_reader.read_csv(table_path);
    auto trimmed_df =
        arrow_table_reader.trim_table(default_configs.MAPPED_PAIR_DICT);
    auto status =
        arrow_table_reader.map_table(trimmed_df, default_configs, mean_dists);

    return std::make_tuple(trimmed_df, arrow_table_reader.focal_,
                           arrow_table_reader.baseline_);
  }

  std::tuple<std::map<double, double>, double>
  eval(const std::string& path, const std::string& table_path,
       const Config& default_configs, int stage, bool is_median) {

    auto [df, focal, baseline] =
        preprocessing(path, table_path, default_configs, is_median);

    arrow::DoubleBuilder abs_error_rate_builder;

    auto gt_dist_chunked_array =
        df->GetColumnByName(default_configs.GT_DIST_NAME);
    auto gt_error_chunked_array =
        df->GetColumnByName(default_configs.GT_ERROR_NAME);
    auto gt_dist_array = std::static_pointer_cast<arrow::DoubleArray>(
        gt_dist_chunked_array->chunk(0));
    auto gt_error_array = std::static_pointer_cast<arrow::DoubleArray>(
        gt_error_chunked_array->chunk(0));

    arrow::Status s1, s2;
    for (auto i = 0; i < gt_dist_array->length(); i++) {
      double abs_error_rate =
          gt_error_array->Value(i) / gt_dist_array->Value(i);
      s1 = abs_error_rate_builder.Append(abs_error_rate);
    }

    std::shared_ptr<arrow::DoubleArray> abs_error_rate_array;
    s2 = abs_error_rate_builder.Finish(&abs_error_rate_array);

    auto abs_error_rate_field =
        arrow::field(default_configs.ABS_ERROR_RATE_NAME, arrow::float64());
    df = df->AddColumn(
               df->num_columns(), abs_error_rate_field,
               std::make_shared<arrow::ChunkedArray>(abs_error_rate_array))
             .ValueOrDie();

    auto max_stage = std::numeric_limits<double>::min();
    for (auto i = 0; i < gt_dist_array->length(); i++) {
      max_stage = std::max(max_stage, gt_dist_array->Value(i));
    }
    int n_stage = static_cast<int>(max_stage / stage);

    std::vector<double> stages;
    for (int i = 0; i < n_stage; i++) {
      stages.push_back(stage + i * stage);
    }

    std::map<double, double> eval_res;
    for (double s : stages) {
      std::vector<double> absolute_error_rates;
      for (int64_t i = 0; i < gt_dist_array->length(); i++) {
        double actual_depth = gt_dist_array->Value(
            i); // Replace with actual depth column if different
        if (actual_depth <= s && actual_depth > s - stage) {
          absolute_error_rates.push_back(gt_error_array->Value(i) /
                                         gt_dist_array->Value(i));
        }
      }
      double mape = 0.0;
      if (!absolute_error_rates.empty()) {
        for (double rate : absolute_error_rates) {
          mape += std::abs(rate);
        }
        mape /= absolute_error_rates.size();
      }
      eval_res[s] = mape;
    }

    int total_bins = eval_res.size();
    int accept = 0;
    for (const auto& [k, v] : eval_res) {
      if ((k <= 1000 && v < 0.01) || (k <= 2000 && v < 0.02)) {
        accept++;
      }
    }

    double acceptance = static_cast<double>(accept) / total_bins;
    return std::make_tuple(eval_res, acceptance);
  }
} // namespace kbd
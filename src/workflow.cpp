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

#include "workflow.h"

#include "optimizer.h"
#include "format.h"
#include "utils.h"

namespace kbd {

  void LinearWorkflow::preprocessing(const std::string& file_path,
                                     const std::string& csv_path,
                                     const Config& config,
                                     const JointSmoothArguments& args) {
    auto table_parser = ArrowTableReader();
    auto df = table_parser.read_csv(csv_path);
    auto trimmed_df = table_parser.trim_table(config.MAPPED_PAIR_DICT);
    auto dist_dict = calculate_mean_value(
        file_path, retrieve_folder_names(file_path), config);
    auto status = table_parser.map_table(trimmed_df, config, dist_dict);

    auto gt_arrow_col = trimmed_df->GetColumnByName(config.GT_DIST_NAME);
    auto est_arrow_col = trimmed_df->GetColumnByName(config.AVG_DISP_NAME);

    auto gt_int64 =
        std::static_pointer_cast<arrow::Int64Array>(gt_arrow_col->chunk(0));
    auto est_double =
        std::static_pointer_cast<arrow::DoubleArray>(est_arrow_col->chunk(0));
    Eigen::Map<const Eigen::Array<int64_t, Eigen::Dynamic, 1>> gt_eigen_array(
        gt_int64->raw_values(), gt_int64->length());
    Eigen::Map<const Eigen::ArrayXd> est_eigen_array(est_double->raw_values(),
                                                     est_double->length());

    focal_ = table_parser.focal_;
    baseline_ = table_parser.baseline_;
    gt_double_ = gt_eigen_array.cast<double>();
    est_double_ = est_eigen_array.cast<double>();
    disjoint_depth_range_ = args.disjoint_depth_range;
    cd_ = args.compensate_dist;
    sf_ = args.scaling_factor;
    apply_global_ = args.apply_global;
    full_kbd_params5x5_.setZero();
    disp_val_max_uint16_ = config.DISP_VAL_MAX_UINT16;
    eval_stage_steps_ = config.EVAL_STAGE_STEPS;
    trimmed_df_ = trimmed_df;
  }

  void LinearWorkflow::optimize() {
    auto linear_kbd_optim = JointLinearSmoothingOptimizer(
        gt_double_, est_double_, focal_, baseline_, disjoint_depth_range_, cd_,
        sf_, apply_global_);

    auto [lm1_, kbd_res_, lm2_] = linear_kbd_optim.run();
  }

  void LinearWorkflow::extend_matrix() {
    double k, delta, b, alpha1, alpha2, beta1, beta2;

    alpha1 = lm1_(0), beta1 = lm1_(1);
    alpha2 = lm2_(0), beta2 = lm2_(1);
    k = kbd_res_(0), delta = kbd_res_(1), b = kbd_res_(2);

    full_kbd_params5x5_ << 1, 0, 0, 1, 0, 1, 0, 0, alpha1, beta1, k, delta, b,
        1, 0, 1, 0, 0, alpha2, beta2, 1, 0, 0, 1, 0;
  }

  std::tuple<Eigen::Array<uint16_t, Eigen::Dynamic, 1>,
             Eigen::Matrix<double, 5, 5>>
  LinearWorkflow::pivot() {
    auto lb = static_cast<double>(disjoint_depth_range_[0]);
    auto ub = static_cast<double>(disjoint_depth_range_[1]);
    auto fb = focal_ * baseline_;

    Eigen::Array<double, 4, 1> extra_range = {lb - cd_, lb, ub, ub + cd_ * sf_};
    Eigen::Array<double, 4, 1> disp_nodes_double = fb / extra_range;
    Eigen::Array<uint16_t, 4, 1> disp_nodes_uint16 =
        (disp_nodes_double * 64).cast<uint16_t>();
    std::sort(disp_nodes_uint16.data(),
              disp_nodes_uint16.data() + disp_nodes_uint16.size());
    Eigen::Array<uint16_t, Eigen::Dynamic, 1> disp_nodes(
        disp_nodes_uint16.size() + 1);
    disp_nodes.head(disp_nodes_uint16.size()) = disp_nodes_uint16;
    disp_nodes(disp_nodes_uint16.size()) = disp_val_max_uint16_;

    auto matrix_param_by_disp = full_kbd_params5x5_.colwise().reverse();

    return std::make_tuple(disp_nodes, matrix_param_by_disp);
  }

  std::tuple<std::map<double, double>, double>
  LinearWorkflow::eval(const Config& config) {
    arrow::DoubleBuilder abs_error_rate_builder;
    auto stage = config.EVAL_STAGE_STEPS;

    auto gt_dist_chunked_array =
        trimmed_df_->GetColumnByName(config.GT_DIST_NAME);
    auto gt_error_chunked_array =
        trimmed_df_->GetColumnByName(config.GT_ERROR_NAME);
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
        arrow::field(config.ABS_ERROR_RATE_NAME, arrow::float64());
    trimmed_df_ =
        trimmed_df_
            ->AddColumn(
                trimmed_df_->num_columns(), abs_error_rate_field,
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
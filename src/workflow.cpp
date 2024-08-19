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

#include "utils.h"
#include "format.h"
#include "ops/modifier.h"
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <cfloat>
#include <unordered_map>

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
    auto avg_z_arrow_col = trimmed_df->GetColumnByName(config.AVG_DIST_NAME);

    auto gt_int64 =
        std::static_pointer_cast<arrow::Int64Array>(gt_arrow_col->chunk(0));
    auto est_double =
        std::static_pointer_cast<arrow::DoubleArray>(est_arrow_col->chunk(0));
    auto avg_z_double =
        std::static_pointer_cast<arrow::DoubleArray>(avg_z_arrow_col->chunk(0));
    Eigen::Map<const Eigen::Array<int64_t, Eigen::Dynamic, 1>> gt_eigen_array(
        gt_int64->raw_values(), gt_int64->length());
    Eigen::Map<const Eigen::ArrayXd> est_eigen_array(est_double->raw_values(),
                                                     est_double->length());
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> avg_z_eigen_array(
        avg_z_double->raw_values(), avg_z_double->length());
    Eigen::ArrayXd error_eigen_array(gt_int64->length());
    for (int i = 0; i < gt_int64->length(); ++i) {
      error_eigen_array(i) = avg_z_double->Value(i) - gt_int64->Value(i);
    }

    focal_ = table_parser.focal_;
    baseline_ = table_parser.baseline_;
    gt_double_ = gt_eigen_array.cast<double>();
    est_double_ = est_eigen_array.cast<double>();
    error_double_ = error_eigen_array;
    avg_z_double_ = avg_z_eigen_array.cast<double>();
    disjoint_depth_range_ = args.disjoint_depth_range;
    cd_ = args.compensate_dist;
    sf_ = args.scaling_factor;
    apply_global_ = args.apply_global;
    full_kbd_params5x5_.setZero();
    disp_val_max_uint16_ = config.DISP_VAL_MAX_UINT16;
    trimmed_df_ = trimmed_df;
    config_ = config;
    args_ = args;

    lazy_compute_ref_z();
  }

  void LinearWorkflow::optimize(OptimizerDiffType diff_type) {
    auto linear_kbd_optim = JointLinearSmoothingOptimizer(
        gt_double_, est_double_, focal_, baseline_, disjoint_depth_range_, cd_,
        sf_, apply_global_);

    linear_kbd_optim.set_optimizer_type(diff_type);

    auto [lm1, kbd_res, lm2] = linear_kbd_optim.run();
    lm1_ = lm1;
    kbd_res_ = kbd_res;
    lm2_ = lm2;
  }

  void LinearWorkflow::line_search(const std::array<int, 2>& search_range,
                                   OptimizerDiffType diff_type) {
    auto lowest_mse = DBL_MAX;
    auto sz = (search_range[1] - search_range[0]) / step_ + 1;
    std::vector<std::array<int, 2>> ranges(sz);
    std::vector<Eigen::Matrix<double, 5, 5>> pms(sz);
    std::vector<Eigen::Vector2d> lm1s(sz);
    std::vector<Eigen::Vector2d> lm2s(sz);
    std::vector<Eigen::Vector3d> kbds(sz);
    std::vector<double> mses(sz);
    std::vector<Eigen::Vector<double, 6>> z_error_rates(sz);

    Eigen::Matrix<double, 5, 5> best_pm;
    std::array<int, 2> best_range{};
    Eigen::Vector<double, 6> best_z_error_rate;

    for (int s = search_range[0]; s <= search_range[1]; s += step_) {
      std::array<int, 2> rg = {s, search_range[1]};
      auto optimizer =
          JointLinearSmoothingOptimizer(gt_double_, est_double_, focal_,
                                        baseline_, rg, cd_, sf_, apply_global_);
      optimizer.set_optimizer_type(diff_type);
      auto [lm1, kbd, lm2] = optimizer.run();
      auto pm = extend_matrix(lm1, kbd, lm2);
      auto [mse, z_error_rate] = evaluate_target(pm, rg, cd_);
      ranges.push_back(rg);
      pms.push_back(pm);
      lm1s.push_back(lm1);
      kbds.push_back(kbd);
      lm2s.push_back(lm2);
      mses.push_back(mse);
      z_error_rates.push_back(z_error_rate);
    }

    // After collecting all results, analyze them based on the given conditions
    for (int i = 0; i < sz; ++i) {
      auto z_er = z_error_rates[i];
      if ((z_er.head(4).array() < 0.02).all() &&
          (z_er.tail(1).array() < 0.04).all()) {
        if (mses[i] < lowest_mse) {
          lowest_mse = mses[i];
          best_range = ranges[i];
          best_pm = pms[i];
          best_z_error_rate = z_error_rates[i];
        }
      }
    }

    // If no suitable mse is found, pick the smallest overall
    if (lowest_mse == DBL_MAX) {
      auto min_iter = std::min_element(mses.begin(), mses.end());
      lowest_mse = *min_iter;
      auto index = std::distance(mses.begin(), min_iter);
      best_range = ranges[index];
      best_pm = pms[index];
      best_z_error_rate = z_error_rates[index];
    }

    best_range_ = best_range;
    best_pm_ = std::move(best_pm);
    best_z_error_rate_ = std::move(best_z_error_rate);

    fmt::print("{:=>50}\n", "");
    fmt::print("Best ranges: {}\n", fmt::join(best_range, ", "));
    fmt::print("{:=>50}\n", "");
    fmt::print("Best z error rate: {}\n", best_z_error_rate);
    fmt::print("{:=>50}\n", "");
  }

  std::tuple<double, Eigen::Matrix<double, 5, 5>, Eigen::Vector2d,
             Eigen::Vector3d, Eigen::Vector2d>
  LinearWorkflow::grid_search::eval_params(int range_start,
                                           double compensate_dist) {
    std::array<int, 2> rng = {range_start, 3000};
    auto jlm = JointLinearSmoothingOptimizer(
        lwf_->gt_double_, lwf_->est_double_, lwf_->focal_, lwf_->baseline_, rng,
        compensate_dist, lwf_->sf_, lwf_->apply_global_, diff_type_);
    auto [lm1, kbd, lm2] = jlm.run();
    auto pm = lwf_->extend_matrix(lm1, kbd, lm2);
    auto [mse, xx] = lwf_->evaluate_target(pm, rng, compensate_dist);
    return {mse, pm, lm1, kbd, lm2};
  }

  void LinearWorkflow::grid_search::optimize_params(
      const std::array<int, 2>& search_range,
      const std::array<double, 2>& cd_range, int max_iter, double tol) {
    auto nelder_mead = NelderMeadOptimizer<double, Eigen::Vector2d>(
        std::bind(&grid_search::objective, this, std::placeholders::_1));
    Eigen::Vector2d ip(2);
    ip << static_cast<double>(search_range[0]), cd_range[0];
    nelder_mead.set_init_simplex(ip);
    Eigen::Vector2d lower_bounds, upper_bounds;
    lower_bounds << static_cast<double>(search_range[0]), cd_range[0];
    upper_bounds << static_cast<double>(search_range[1]), cd_range[1];
    nelder_mead.set_bounds(lower_bounds, upper_bounds);
    auto res = nelder_mead.optimize();
    optim_res_ = std::move(res);
  }

  double LinearWorkflow::grid_search::objective(const Eigen::Vector2d& params) {
    auto [mse, pm, lm1, kbd, lm2] =
        this->eval_params(static_cast<int>(params(0)), params(1));
    return mse;
  }

  std::tuple<Eigen::Matrix<double, 5, 5>, int, double>
  LinearWorkflow::grid_search::get_results() {
    auto best_rng_start = static_cast<int>(optim_res_(0));
    auto best_cd = optim_res_(1);
    auto [best_mse, best_pm, lm1, kbd, lm2] =
        this->eval_params(best_rng_start, best_cd);
    fmt::print("Optimization successful.\n");
    fmt::print("Optimized range start: {}\n", best_rng_start);
    fmt::print("Optimized compensate distance: {}\n", best_cd);
    fmt::print("Minimum MSE: {}\n", best_mse);

    return {best_pm, best_rng_start, best_cd};
  }

  void LinearWorkflow::extend_matrix() {
    double k, delta, b, alpha1, alpha2, beta1, beta2;

    alpha1 = lm1_(0), beta1 = lm1_(1);
    alpha2 = lm2_(0), beta2 = lm2_(1);
    k = kbd_res_(0), delta = kbd_res_(1), b = kbd_res_(2);

    full_kbd_params5x5_ << 1, 0, 0, 1, 0, 1, 0, 0, alpha1, beta1, k, delta, b,
        1, 0, 1, 0, 0, alpha2, beta2, 1, 0, 0, 1, 0;
  }

  Eigen::Matrix<double, 5, 5>
  LinearWorkflow::extend_matrix(const Eigen::Vector2d& lm1,
                                const Eigen::Vector3d& kbd,
                                const Eigen::Vector2d& lm2) {
    double k, delta, b, alpha1, alpha2, beta1, beta2;
    alpha1 = lm1(0), beta1 = lm1(1);
    alpha2 = lm2(0), beta2 = lm2(1);
    k = kbd(0), delta = kbd(1), b = kbd(2);

    Eigen::Matrix<double, 5, 5> mat;

    mat << 1, 0, 0, 1, 0, 1, 0, 0, alpha1, beta1, k, delta, b, 1, 0, 1, 0, 0,
        alpha2, beta2, 1, 0, 0, 1, 0;

    return mat;
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

  std::tuple<Eigen::Array<uint16_t, Eigen::Dynamic, 1>,
             Eigen::Matrix<double, 5, 5>>
  LinearWorkflow::pivot(const Eigen::Matrix<double, 5, 5>& m,
                        const std::array<int, 2>& rng, double cd) {
    auto lb = static_cast<double>(rng[0]);
    auto ub = static_cast<double>(rng[1]);
    auto fb = focal_ * baseline_;

    Eigen::Array<double, 4, 1> extra_range = {lb - cd, lb, ub, ub + cd * sf_};
    Eigen::Array<double, 4, 1> disp_nodes_double = fb / extra_range;
    Eigen::Array<uint16_t, 4, 1> disp_nodes_uint16 =
        (disp_nodes_double * 64).cast<uint16_t>();
    std::sort(disp_nodes_uint16.data(),
              disp_nodes_uint16.data() + disp_nodes_uint16.size());
    Eigen::Array<uint16_t, Eigen::Dynamic, 1> disp_nodes(
        disp_nodes_uint16.size() + 1);
    disp_nodes.head(disp_nodes_uint16.size()) = disp_nodes_uint16;
    disp_nodes(disp_nodes_uint16.size()) = disp_val_max_uint16_;

    auto matrix_param_by_disp = m.colwise().reverse();

    return std::make_tuple(disp_nodes, matrix_param_by_disp);
  }

  std::tuple<Eigen::Array<uint16_t, Eigen::Dynamic, 1>,
             Eigen::Matrix<double, 5, 5>>
  LinearWorkflow::export_default() const {
    const double lb = 600;
    const double ub = 3000;
    const double fb = focal_ * baseline_;
    const double cd = 100;

    Eigen::Matrix<double, 5, 5> m;
    m << 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
        0;

    Eigen::Array<double, 4, 1> extra_range = {lb - cd, lb, ub, ub + cd * sf_};
    Eigen::Array<double, 4, 1> disp_nodes_double = fb / extra_range;
    Eigen::Array<uint16_t, 4, 1> disp_nodes_uint16 =
        (disp_nodes_double * 64).cast<uint16_t>();
    std::sort(disp_nodes_uint16.data(),
              disp_nodes_uint16.data() + disp_nodes_uint16.size());
    Eigen::Array<uint16_t, Eigen::Dynamic, 1> disp_nodes(
        disp_nodes_uint16.size() + 1);
    disp_nodes.head(disp_nodes_uint16.size()) = disp_nodes_uint16;
    disp_nodes(disp_nodes_uint16.size()) = disp_val_max_uint16_;

    auto matrix_param_by_disp = m.colwise().reverse();

    return std::make_tuple(disp_nodes, matrix_param_by_disp);
  }

  std::tuple<std::map<double, double>, double> LinearWorkflow::eval() {
    arrow::DoubleBuilder abs_error_rate_builder;
    auto stage = config_.EVAL_STAGE_STEPS;

    auto gt_dist_chunked_array =
        trimmed_df_->GetColumnByName(config_.GT_DIST_NAME);
    auto gt_error_chunked_array =
        trimmed_df_->GetColumnByName(config_.GT_ERROR_NAME);
    auto gt_dist_array = std::static_pointer_cast<arrow::Int64Array>(
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
        arrow::field(config_.ABS_ERROR_RATE_NAME, arrow::float64());
    trimmed_df_ =
        trimmed_df_
            ->AddColumn(
                trimmed_df_->num_columns(), abs_error_rate_field,
                std::make_shared<arrow::ChunkedArray>(abs_error_rate_array))
            .ValueOrDie();
    auto abs_error_rate =
        trimmed_df_->GetColumnByName(config_.ABS_ERROR_RATE_NAME);
    auto max_stage = std::numeric_limits<double>::min();
    for (auto i = 0; i < gt_dist_array->length(); i++) {
      max_stage =
          std::max(max_stage, static_cast<double>(gt_dist_array->Value(i)));
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
      if ((k <= 1000 && v < 0.02) || (k <= 2000 && v < 0.04)) {
        accept++;
      }
    }
    double acceptance = static_cast<double>(accept) / total_bins;
    return std::make_tuple(eval_res, acceptance);
  }

  bool LinearWorkflow::pass_or_not() {
    std::unordered_map<int, double> metric_thresholds;
    for (int i = 0; i < 4; ++i) {
      metric_thresholds[metric_points_[i]] = 0.02;
    }
    for (int i = 4; i < metric_points_.size(); ++i) {
      metric_thresholds[metric_points_[i]] = 0.04;
    }

    int sz = gt_double_.size();
    for (int i = 0; i < sz; ++i) {
      auto z_val = gt_double_(i);
      auto error_val = error_double_(i);
      if (metric_thresholds.find(z_val) != metric_thresholds.end()) {
        double err = std::abs(error_val / z_val);
        if (err > metric_thresholds[z_val]) {
          return false;
        }
      }
    }
    return true;
  }

  bool LinearWorkflow::ratio_evaluate(double alpha, int min_offset) {
    for (int i = 0; i < gt_double_.size(); ++i) {
      auto z_true = gt_double_(i);
      if (z_true < min_offset)
        continue;

      double d_value = focal_ * baseline_ / z_true;
      double err_rate_value = std::abs(error_double_(i) / z_true);
      double ratio_value = 1 / (1 - alpha * (1 / d_value)) - 1;

      if (ratio_value - err_rate_value < 0) {
        return false;
      }
    }
    return true;
  }

  bool LinearWorkflow::first_check(double max_thr, double mean_thr) {
    auto sz = gt_double_.size();
    Eigen::VectorXd err_rate(sz);
    for (int i = 0; i < sz; ++i) {
      err_rate(i) = std::abs(error_double_(i) / gt_double_(i));
    }
    return (err_rate.maxCoeff() < max_thr && err_rate.mean() < mean_thr);
  }

  bool LinearWorkflow::final_check(const Eigen::Matrix<double, 5, 5>& pm,
                                   const std::array<int, 2>& range, double cd,
                                   double weights_factor) {
    auto avg_depth = focal_ * baseline_ / est_double_;
    ndArray<double> m_row_major = avg_depth;
    auto pred =
        ops::modify_linear(m_row_major, focal_, baseline_, pm, range, cd, sf_);
    auto gt_error = (error_double_ / gt_double_).abs();
    Eigen::ArrayXd kbd_error = ((gt_double_ - pred.array()) / gt_double_).abs();

    Eigen::ArrayXd max_error = Eigen::Map<Eigen::ArrayXd>(
        args_.thresholds.data(), args_.thresholds.size());

    std::unordered_map<double, int> index_map;
    for (int i = 0; i < gt_double_.size(); ++i) {
      index_map[gt_double_(i)] = i;
    }

    Eigen::ArrayXd kbd_error_slice(metric_points_.size());
    int slice_index = 0;
    for (double point : metric_points_) {
      auto it = index_map.find(point);
      if (it != index_map.end()) {
        kbd_error_slice(slice_index++) = kbd_error(it->second);
      }
    }

    if ((kbd_error_slice < max_error).all())
      final_pass_ = true;

    Eigen::ArrayXd sample_weights(gt_double_.size());
    for (int i = 0; i < gt_double_.size(); ++i) {
      sample_weights(i) =
          (std::find(metric_points_.begin(), metric_points_.end(),
                     gt_double_(i)) != metric_points_.end())
              ? weights_factor
              : 1.0;
    }
    auto before_mse =
        weighted_mse<double>(gt_double_, avg_depth, sample_weights);
    auto after_mse =
        weighted_mse<double>(gt_double_, pred.array(), sample_weights);
    return after_mse < before_mse;
  }

  std::tuple<double, Eigen::Vector<double, 6>> LinearWorkflow::evaluate_target(
      const Eigen::Matrix<double, 5, 5>& param_matrix,
      const std::array<int, 2>& rg,
      double cd) {
    auto z_after = ops::modify_linear<double>(ref_z_, focal_, baseline_,
                                              param_matrix, rg, cd, sf_);
    auto z_error_rate =
        ((z_after.array() - ref_z_.array()) / ref_z_.array()).abs();
    auto mse = (z_after - ref_z_).array().square().mean();

    return {mse, z_error_rate};
  }

  void LinearWorkflow::lazy_compute_ref_z() {
    ndArray<double> z_array(metric_points_.size(), 1);
    z_array << metric_points_[0], metric_points_[1], metric_points_[2],
        metric_points_[3], metric_points_[4], metric_points_[5];

    ref_z_ = z_array;
  }

  double LinearWorkflow::get_focal_val() const { return this->focal_; }
  double LinearWorkflow::get_baseline_val() const { return this->baseline_; }

} // namespace kbd
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

#ifndef KBD_CONFIG_H
#define KBD_CONFIG_H

#include "shared.h"
#include <array>
#include <map>

namespace kbd {

  struct Config {
    explicit Config(const std::string& mode = "N9") : mode_(mode) {
      initialize();
    }
    explicit Config(std::string&& mode = "N9") : mode_(std::move(mode)) {
      initialize();
    }
    uint16_t DISP_VAL_MAX_UINT16 = 32767;
    std::string SUBFIX = "DEPTH/raw";
    std::string CAMPARAM_NAME = "camparam.txt";
    int H, W;
    float EPSILON = 1e-6;
    int EVAL_STAGE_STEPS = 200;
    double EVAL_WARNING_RATE = 0.5f;
    double FOCAL_MULTIPLIER = 1.0f;
    std::vector<int> ANCHOR_POINT;

    std::string AVG_DIST_NAME = "avg_depth_50x50_anchor";
    std::string AVG_DISP_NAME = "avg_disp_50x50_anchor";
    std::string MEDIAN_DIST_NAME = "median_depth_50x50_anchor";
    std::string MEDIAN_DISP_NAME = "median_disp_50x50_anchor";
    std::string ABS_ERROR_RATE_NAME = "absolute_error_rate";
    std::string GT_DIST_NAME = "actual_depth";
    std::string GT_ERROR_NAME = "absolute_error";
    std::string FOCAL_NAME = "focal";
    std::string BASELINE_NAME = "baseline";
    std::string BASE_OUTPUT_JSON_FILE_NAME_PREFIX =
        "segmented_linear_KBD_params";
    std::map<std::string, std::string> MAPPED_PAIR_DICT = {
        {"DISTANCE(mm)", "actual_depth"},
        {"Camera_Focal", "focal"},
        {"Camera_Baseline", "baseline"},
        {"Absolute_error/mm", "absolute_error"},
    };

    void initialize() {
      if (mode_ == "N9") {
        H = 480, W = 640;
        FOCAL_MULTIPLIER = 1.6f;
        ANCHOR_POINT = {H / 2, W / 2};
      } else if (mode_ == "M1F") {
        H = 400, W = 640;
        FOCAL_MULTIPLIER = 1.0f;
        ANCHOR_POINT = {H / 2, W / 2};
      } else {
        std::runtime_error(
            "Unsupported mode! Please recheck the requirements!");
      }
    }

  private:
    std::string mode_;
  };

  struct JointSmoothArguments {
    explicit JointSmoothArguments(const std::string& mode = "N9")
        : mode_(mode) {
      initialize();
    }
    explicit JointSmoothArguments(std::string&& mode = "N9")
        : mode_(std::move(mode)) {
      initialize();
    }
    std::array<int, 2> disjoint_depth_range = {600, 3000};
    std::array<int, 6> metric_points = {300, 500, 600, 1000, 1500, 2000};
    std::array<double, 6> thresholds;
    double near_thr;
    double far_thr;
    double compensate_dist = 200;
    double scaling_factor = 10;
    bool apply_global = false;

    void initialize() {
      if (mode_ == "N9") {
        thresholds = {0.02, 0.02, 0.02, 0.02, 0.04, 0.04};
        near_thr = 0.02, far_thr = 0.04;
      } else if (mode_ == "M1F") {
        thresholds = {0.02, 0.02, 0.02, 0.02, 0.02, 0.02};
        near_thr = 0.02, far_thr = 0.02;
      } else {
        std::runtime_error(
            "Unsupported mode! Please recheck the requirements!");
      }
    }

  private:
    std::string mode_;
  };
} // namespace kbd

#endif // KBD_CONFIG_H

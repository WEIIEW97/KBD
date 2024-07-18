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
    uint16_t DISP_VAL_MAX_UINT16 = 32767;
    std::string SUBFIX = "DEPTH/raw";
    std::string CAMPARAM_NAME = "camparam.txt";
    int H = 480;
    int W = 640;
    float EPSILON = 1e-6;
    int EVAL_STAGE_STEPS = 200;
    double EVAL_WARNING_RATE = 0.5f;

    std::vector<int> ANCHOR_POINT = {H / 2, W / 2};

    std::string AVG_DIST_NAME = "avg_depth_50x50_anchor";
    std::string AVG_DISP_NAME = "avg_disp_50x50_anchor";
    std::string MEDIAN_DIST_NAME = "median_depth_50x50_anchor";
    std::string MEDIAN_DISP_NAME = "median_disp_50x50_anchor";
    std::string ABS_ERROR_RATE_NAME = "absolute_error_rate";
    std::string GT_DIST_NAME = "actual_depth";
    std::string GT_ERROR_NAME = "absolute_error";
    std::string FOCAL_NAME = "focal";
    std::string BASELINE_NAME = "baseline";
    std::string BASE_OUTPUT_JSON_FILE_NAME_PREFIX = "segmented_linear_KBD_params";
    std::map<std::string, std::string> MAPPED_PAIR_DICT = {
        {"距离(mm)", "actual_depth"},
        {"相机焦距", "focal"},
        {"相机基线", "baseline"},
        {"绝对误差/mm", "absolute_error"},
    };
  };

  struct JointSmoothArguments {
    std::array<int, 2> disjoint_depth_range = {600, 3000};
    double compensate_dist = 400;
    double scaling_factor = 10;
    bool apply_global = false;
  };
} // namespace kbd

#endif // KBD_CONFIG_H

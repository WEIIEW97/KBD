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
    int H = 480;
    int W = 640;
    float EPSILON = 1e-6;

    std::vector<int> anchor_point = {H/2, W/2};

    std::string AVG_DIST_NAME = "avg_depth_50x50_anchor";
    std::string AVG_DISP_NAME = "avg_disp_50x50_anchor";
    std::string MEDIAN_DIST_NAME = "median_depth_50x50_anchor";
    std::string MEDIAN_DISP_NAME = "median_disp_50x50_anchor";
    std::string GT_DIST_NAME = "actual_depth";
    std::string GT_ERROR_NAME = "absolute_error";
    std::string FOCAL_NAME = "focal";
    std::string BASELINE_NAME = "baseline";
    std::map<std::string, std::string> MAPPED_PAI_DICT = {
        {"距离(mm)", "actual_depth"},
        {"相机焦距", "focal"},
        {"相机基线", "baseline"},
        {"绝对误差/mm", "absolute_error"},
    };
  };
}

#endif // KBD_CONFIG_H
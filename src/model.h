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

#pragma once

#include <Eigen/Dense>

namespace kbd {
  template <typename T>
  Eigen::Array<T, Eigen::Dynamic, 1>
  basic_model(const Eigen::Array<T, Eigen::Dynamic, 1>& disp, T focal,
              T baseline, T k, T delta, T b) {
    return k * focal * baseline / (disp + delta) + b;
  }
} // namespace kbd
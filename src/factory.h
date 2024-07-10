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

#include "table.h"
#include "utils.h"

namespace kbd {
  std::tuple<std::shared_ptr<arrow::Table>, double, double>
  preprocessing(const std::string& path, const std::string& table_path,
                const Config& default_configs, bool is_median);
  std::tuple<std::map<double, double>, double>
  eval(const std::string& path, const std::string& table_path,
       const Config& default_configs, int stage, bool is_median);
} // namespace kbd
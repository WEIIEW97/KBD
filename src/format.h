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
#include <fmt/core.h>
#include <fmt/format.h>

// Custom formatter for Eigen::Array
namespace fmt {
  template <typename T, int Rows, int Cols>
  struct formatter<Eigen::Array<T, Rows, Cols>> {
    // Parses format specifications from the format string
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
      // No specific format options are needed, so just return the end iterator.
      return ctx.end();
    }

    // Formats the Eigen::Array using the given format context
    template <typename FormatContext>
    auto format(const Eigen::Array<T, Rows, Cols>& array, FormatContext& ctx)
        -> decltype(ctx.out()) {
      // Start formatting with a bracket
      auto out = ctx.out();
      fmt::format_to(out, "[");

      // Iterate over the elements of the array
      for (int i = 0; i < array.size(); ++i) {
        fmt::format_to(out, "{}", array(i));
        if (i < array.size() - 1) {
          fmt::format_to(out, ", ");
        }
      }

      // End formatting with a bracket
      return fmt::format_to(out, "]");
    }
  };
} // namespace fmt
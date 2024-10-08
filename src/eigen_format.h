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
#include <fmt/ostream.h>
#include <regex>
#include <iostream>
#include <cstring>
// #include <fmt/ranges.h>

// Custom formatter for Eigen::Array
namespace fmt {
  template <typename Derived>
  struct fmt::formatter<Eigen::MatrixBase<Derived>> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
      auto it = ctx.begin(), end = ctx.end();
      // No format specifiers are expected, check for a literal '}' or end.
      if (it != end && *it != '}') {
        throw fmt::format_error("invalid format");
      }
      return it;
    }

    template <typename FormatContext>
    auto format(const Eigen::MatrixBase<Derived>& mat, FormatContext& ctx)
        -> decltype(ctx.out()) {
      format_to(ctx.out(), "[");
      for (int i = 0; i < mat.size(); ++i) {
        if (i > 0)
          format_to(ctx.out(), ", ");
        format_to(ctx.out(), "{}", mat(i));
      }
      return format_to(ctx.out(), "]");
    }
  };
} // namespace fmt
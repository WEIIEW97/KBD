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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace kbd {
  template <typename DataType, typename MaskType>
  Eigen::Array<DataType, Eigen::Dynamic, 1>
  mask_out_array(const Eigen::Array<DataType, Eigen::Dynamic, 1>& arr,
                 const Eigen::Array<MaskType, Eigen::Dynamic, 1>& mask) {
    std::vector<DataType> holder;
    for (int i = 0; i < mask.size(); ++i) {
      if (mask(i)) {
        holder.push_back(arr(i));
      }
    }
    Eigen::Array<DataType, Eigen::Dynamic, 1> masked_data(holder.size());
    for (int i = 0; i < holder.size(); ++i) {
      masked_data(i) = holder[i];
    }
    return masked_data;
  }

  template <typename DataType, typename MaskType>
  Eigen::Matrix<DataType, Eigen::Dynamic, 1>
  mask_out_vector(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>& arr,
                  const Eigen::Matrix<MaskType, Eigen::Dynamic, 1>& mask) {
    std::vector<DataType> holder;
    for (int i = 0; i < mask.size(); ++i) {
      if (mask(i)) {
        holder.push_back(arr(i));
      }
    }

    // Creating a dynamic-sized Eigen vector to return
    Eigen::Matrix<DataType, Eigen::Dynamic, 1> masked_data(holder.size());
    for (int i = 0; i < holder.size(); ++i) {
      masked_data(i) = holder[i];
    }
    return masked_data;
  }
} // namespace kbd
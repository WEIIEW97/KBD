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

#include <Eigen/Core>
#include <array>
#include <iostream>

#include "../src/eigen_utils.h"

using namespace Eigen;
using namespace std;


int main() {
  array<double, 20> data = {0};
  int start_v = 300;
  const int stride = 150;
  const int size = 20;
  for (int i = 0; i < size; ++i) {
    data[i] = start_v + (i - 1) * stride;
  }

  for(const auto& v : data) {
    cout << v << "\n";
  }

  auto eigen_data_map = Map<Array<double, 20, 1>>(data.data(), data.size());
  Array<double, 20, 1> eigen_data = eigen_data_map;
  std::array<int, 2> range = {600, 1500};
  Eigen::Array<bool, Eigen::Dynamic, 1> mask =
      (eigen_data > range[0]) && (eigen_data < range[1]);
  auto masked_data = eigen_data.select(mask, 0);

  cout << "original data is : " << eigen_data << endl;
  cout << "bool mask data is : " << mask << endl;
  cout << "after masking data is: " << kbd::mask_out_array<double, bool>(eigen_data, mask) << endl;

  return 0;
}
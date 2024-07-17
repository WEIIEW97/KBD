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

#include "../src/eigen_io.h"
#include "../src/utils.h"
#include <string>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
  Eigen::Matrix<uint16_t, 5, 5> mat;
  mat.setRandom();
  std::cout << "write : " << "\n";
  std::cout << mat << std::endl;
  auto cwd = fs::current_path(); // note that this is the binary path
  auto home_path = cwd.parent_path().string();
  std::string test_matrix_save_path = home_path + "/test/test_depth.raw";

  kbd::write_binary(test_matrix_save_path, mat);

  Eigen::Matrix<uint16_t, -1, -1> mat_r;
  kbd::read_binary(test_matrix_save_path, mat_r);
  std::cout << "read :" << "\n";
  std::cout << mat_r << std::endl;

  std::string another_fixed_path = "/home/william/Codes/KBD/data/N09ASH24DH0050/image_data/155_N09ASH24DH0050_2024_07_09_11_30_35/DEPTH/raw/Depth-2024-07-09-11-30-35-754-1-000356-1720495835391476.raw";
  auto mat_f = kbd::load_raw<uint16_t>(another_fixed_path, 480, 640);
  std::cout << "fixed :" << "\n";
  std::cout << mat_f << std::endl;
  return 0;
}
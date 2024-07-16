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

 #include "../src/workflow.h"
 #include <iostream>

 int main() {
  const std::string root_path = "/home/william/Codes/KBD/data/N09ASH24DH0050";
  const std::string csv_path = "/home/william/Codes/KBD/data/N09ASH24DH0050/depthquality_2024-07-09.csv";
  const std::string file_path = "/home/william/Codes/KBD/data/N09ASH24DH0050/image_data";
  kbd::Config default_configs = kbd::Config();
  kbd::JointSmoothArguments args = kbd::JointSmoothArguments();

  kbd::LinearWorkflow workflow;

  workflow.preprocessing(file_path, csv_path, default_configs, args);
  auto [eval_res, acceptance] = workflow.eval(default_configs);
  std::cout << "acceptance rate: " << acceptance << std::endl;

  workflow.optimize();
  workflow.extend_matrix();
  workflow.pivot();
  
  auto [disp_nodes, param_matrix] = workflow.pivot();
  std::cout << disp_nodes << std::endl;
  std::cout << param_matrix << std::endl;

  return 0;
 }
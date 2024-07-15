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

#include "../src/utils.h"
#include <Eigen/Core>
#include <fstream>
#include <string>

int main() {
    std::string test_json_save_path = "/home/william/Codes/KBD/test/test.json";
    kbd::json J;
    Eigen::Matrix3d mat = Eigen::Matrix3d::Random();
    std::vector<uint16_t> disp_nodes = {1,2,3,4,5};

    for (auto i : disp_nodes) {
      J["disp_nodes"].push_back(i);
    }

    for (int i = 0; i < mat.rows(); ++i) {
        kbd::json row = kbd::json::array();
        for (int j = 0; j < mat.cols(); ++j) {
            row.push_back(mat(i,j));
        }
        J["kbd_params"].push_back(row);
    }
    
    std::cout << "matrix is: " << "\n";
    std::cout << mat << std::endl; 

    std::ofstream file(test_json_save_path);
    file << J.dump(4);
    file.close();

    return 0;
}
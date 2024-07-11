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

#include "../src/optimizer.h"
#include "../src/model.h"
#include <iostream>

 int main() {
    Eigen::ArrayXd gt(3);
    gt << 1.0, 2.0, 3.0;
    Eigen::ArrayXd est(3);
    est << 1.1, 2.1, 3.1;
    double focal = 1.0;
    double baseline = 1.0;
    double k = 0.49;
    double b = 0.444;
    double delta = 0.02;

//    kbd::CeresAutoDiffOptimizer optimizer(gt, est, focal, baseline);
//    optimizer.run();

    auto res = kbd::basic_model(est, focal, baseline, k, delta, b);
    std::cout << res << std::endl;

    return 0;
 }
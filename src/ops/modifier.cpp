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

#include "modifier.h"
#include "multi_threads.h"

namespace kbd {
  namespace ops {

    void parallel_copy(const std::string& src, const std::string& dst,
                       const Config& configs) {
      auto subfix = configs.SUBFIX;
      auto camparam_name = configs.CAMPARAM_NAME;

      auto folders = retrieve_folder_names(src);
      ThreadPool pool(std::thread::hardware_concurrency());
#if 0
#pragma omp parallel for
#endif
      for (const auto& folder : folders) {
        auto source_path = src + "/" + folder + "/" + subfix;
        auto destination_path = dst + "/" + folder + "/" + subfix;
        auto cam_source = src + "/" + folder + "/" + camparam_name;
        auto cam_dest = dst + "/" + folder + "/" + camparam_name;

        pool.enqueue(copy_files_in_directory, source_path, destination_path);
        pool.enqueue(
            [](const std::string& src, const std::string& dst) {
              fs::copy(src, dst, fs::copy_options::overwrite_existing);
            },
            cam_source, cam_dest);
      }

      fmt::print("Copying done! ... \n");
    }

    void linear_transform_kernel(
        const std::string& path, int h, int w, double f, double b,
        const Eigen::Matrix<double, 5, 5>& params_matrix, double cd, double sf,
        const std::array<int, 2>& range) {
      for (const auto& entry : fs::directory_iterator(path)) {
        auto p = entry.path().string();
        auto raw = load_raw<uint16_t>(p, h, w);
        auto depth =
            modify_linear<uint16_t>(raw, f, b, params_matrix, range, cd, sf);
        save_raw<uint16_t>(p, depth);
      }
    }

    void parallel_transform(const std::string& path,
                            const Eigen::Matrix<double, 5, 5>& params_matrix,
                            double focal, double baseline,
                            const Config& configs,
                            const JointSmoothArguments& arguments) {
      auto range = arguments.disjoint_depth_range;
      auto cd = arguments.compensate_dist;
      auto sf = arguments.scaling_factor;
      auto h = configs.H;
      auto w = configs.W;
      auto subfix = configs.SUBFIX;

      auto folders = retrieve_folder_names(path);
      ThreadPool pool(std::thread::hardware_concurrency());
#if 0
#pragma omp parallel for
#endif
      for (const auto& folder : folders) {
        auto full_path = fs::path(path) / folder / subfix;
        // pool.enqueue(linear_transform_kernel, full_path, h, w, focal,
        // baseline,
        //              params_matrix, cd, sf, range);
        pool.enqueue([=, &params_matrix] { // Capture by value, but
                                           // params_matrix by reference
          linear_transform_kernel(full_path.string(), h, w, focal, baseline,
                                  params_matrix, cd, sf, range);
        });
      }
      fmt::print("Transformating data done ...\n");
    }
  } // namespace ops
} // namespace kbd
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

#include "multi_threads.h"
#include "../utils.h"

#include <fmt/core.h>

namespace kbd {
  namespace ops {
    ThreadPool::ThreadPool(size_t threads) : stop(false) {
      for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
          for (;;) {
            std::function<void()> task;

            {
              std::unique_lock<std::mutex> lock(this->queue_mutex);
              this->condition.wait(
                  lock, [this] { return this->stop || !this->tasks.empty(); });
              if (this->stop && this->tasks.empty())
                return;
              task = std::move(this->tasks.front());
              this->tasks.pop();
            }

            task();
          }
        });
    }

    ThreadPool::~ThreadPool() {
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
      }
      condition.notify_all();
      for (std::thread& worker : workers)
        worker.join();
    }

    void parallel_copy(const std::string& src, const std::string& dst,
                       const Config& configs) {
      auto subfix = configs.SUBFIX;
      auto camparam_name = configs.CAMPARAM_NAME;

      auto folders = retrieve_folder_names(src);
      ThreadPool pool(std::thread::hardware_concurrency());

      for (const auto& folder : folders) {
        auto source_path = src + "/" + folder + subfix;
        auto destination_path = dst + "/" + folder + subfix;
        auto cam_source = src + "/" + folder + "/" + camparam_name;
        auto cam_dest = dst + "/" + folder + "/" + camparam_name;

        pool.enqueue(copy_files_in_directory, source_path, destination_path);
        pool.enqueue(
            [](const std::string& src, const std::string& dst) {
              fs::copy(src, dst, fs::copy_options::overwrite_existing);
            },
            cam_source, cam_dest);
      }

      fmt::print("Copying done!\n");
    }

  } // namespace ops
} // namespace kbd
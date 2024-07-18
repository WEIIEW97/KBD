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

namespace kbd {
  namespace ops {
    ThreadPool::ThreadPool(size_t threads) : stop_(false) {
      for (size_t i = 0; i < threads; ++i)
        workers_.emplace_back([this] {
          for (;;) {
            std::function<void()> task;

            {
              std::unique_lock<std::mutex> lock(this->queue_mutex_);
              this->condition_.wait(lock, [this] {
                return this->stop_ || !this->tasks_.empty();
              });
              if (this->stop_ && this->tasks_.empty())
                return;
              task = std::move(this->tasks_.front());
              this->tasks_.pop();
            }

            task();
          }
        });
    }

    ThreadPool::~ThreadPool() {
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
      }
      condition_.notify_all();
      for (std::thread& worker : workers_)
        worker.join();
    }
  } // namespace ops
} // namespace kbd
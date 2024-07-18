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


#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>

namespace kbd {
  namespace ops {

    class ThreadPool {
    public:
      ThreadPool(size_t threads);
      ~ThreadPool();

      template <class F, class... Args>
      auto enqueue(F&& f, Args&&... args)
          -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);

          if (stop_)
            throw std::runtime_error("enqueue on stopped ThreadPool");

          tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return res;
      }

    private:
      std::vector<std::thread> workers_;
      std::queue<std::function<void()>> tasks_;
      std::mutex queue_mutex_;
      std::condition_variable condition_;
      bool stop_;
    };
  } // namespace ops
} // namespace kbd
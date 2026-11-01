/**
 * Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
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
#ifndef K2_CSRC_THREAD_POOL_H_
#define K2_CSRC_THREAD_POOL_H_

#include <condition_variable>  // NOLINT
#include <functional>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <vector>

#include "k2/csrc/log.h"

namespace k2 {

class ThreadPool {
 public:
  using TaskFunc = std::function<void()>;

  /* Create a pool with given number of threads.
   *
   * @param [in] num_threads  The number of threads in the pool. If it is <= 0,
   *                          the number of threads in the pool is set to
   *                          `std::thread::hardware_concurrency()`.
   */
  explicit ThreadPool(int32_t num_threads);

  // It has un-copyable members of type `std::mutex` and
  // `std::condition_variable` so this class is not copyable.

  // The destructor waits for all threads in the pool to exit.
  ~ThreadPool();

  // return the number of threads in the pool
  int32_t GetNumThreads() const {
    return static_cast<int32_t>(threads_.size());
  }

  /* Insert a task into the queue. It is executed immediately
   * if there are currently idle threads. Otherwise, it is saved
   * in the queue and waits for an idle thread to be free to pop it from
   * the queue and to execute it.
   *
   * @param [in] task   The task to be run. It should be convertible to
   *                    `std::function<void()>`.
   */
  template <typename Lambda>
  void SubmitTask(Lambda task) {
    K2_CHECK_GT(threads_.size(), 0u);
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.emplace(static_cast<TaskFunc>(task));
    not_empty_cond_.notify_one();
    finished_ = false;
  }

  /* The caller is blocked if `tasks_` is not empty.
   * If `tasks_` is empty, it returns immediately.
   */
  void WaitAllTasksFinished();

  /* Non-blocking quiescence query: true when `tasks_` is empty and no task is
   * running (the same predicate `WaitAllTasksFinished()` waits for). Used by
   * callers that must not block while background tasks may still need a
   * signal from the very thread that is leaving the scope.
   */
  bool AllTasksFinished();

 private:
  /* Remove tasks from `task_` and process them.
   *
   * If `keep_running_` is false, the function will return
   * after processing the current task.
   *
   * If the task queue is empty, the caller is blocked.
   */
  void ProcessTasks();

 private:
  // new tasks are added to this queue
  std::queue<TaskFunc> tasks_;

  std::vector<std::thread> threads_;
  std::mutex mutex_;

  // When the last task is removed and processed from `tasks_`,
  // `empty_cond_` is signaled.
  std::condition_variable empty_cond_;

  // When a new task is added to `tasks_`,
  // `not_empty_cond_` is signaled.
  std::condition_variable not_empty_cond_;

  // Set it to false in the destructor to
  // ask threads to not process tasks in the
  // waiting queue
  bool keep_running_ = true;

  // Set it to true when the task queue is empty
  bool finished_ = true;

  // Whenever a thread is about to process a task,
  // the counter is incremented
  //
  // Whenever a thread finishes processing a task,
  // the counter is decremented.
  int32_t running_counter_ = 0;
};

/* Get a pointer to the thread pool visible in the caller's scope.
 *
 * Priority of resolution:
 *   1. If the caller thread is inside a `ScopedThreadPoolOverride` scope,
 *      the pool owned by that scope is returned (per-scope pool).
 *   2. Otherwise, a process-wide global pool is returned (created lazily
 *      with 2 threads and never freed — upstream behavior).
 *
 * A pool task thread must never obtain a pool this way: k2 phase tasks never
 * call GetThreadPool() (only caller-side code in intersect/connect does), so
 * calling it from a task thread is a program invariant violation and fails
 * fast with K2_LOG(FATAL) instead of silently reaching across pools and
 * waiting on tasks owned by another execution scope.
 *
 * The returned pointer is NOT owned by the caller and
 * it must not be freed by the caller.
 */
ThreadPool *GetThreadPool();

/* True on threads that are owned by a `ThreadPool` and that execute
 * `ProcessTasks()` for it. Callers that take external execution leases
 * (e.g. lattifai-engine's search execution contexts) use this to reject
 * dangerous re-entry from task threads. */
bool IsThreadPoolTaskThread();

/* RAII: while in scope on the current thread, `GetThreadPool()` returns
 * `pool` (which must outlive the scope) instead of the global fallback
 * pool. The previously overridden pool, if any, is restored on
 * destruction.
 *
 * This must be used from the caller thread, never from a pool task
 * thread; the constructor fails fast on a task thread for the same reason
 * as `GetThreadPool()`. Enforcing the caller-side invariant is the lease's
 * job (see lattifai-engine `search_execution.h`).
 */
class ScopedThreadPoolOverride {
 public:
  explicit ScopedThreadPoolOverride(ThreadPool *pool);

  // Restores the previous override; the pool itself is owned elsewhere.
  ~ScopedThreadPoolOverride();

  ScopedThreadPoolOverride(const ScopedThreadPoolOverride &) = delete;
  ScopedThreadPoolOverride &operator=(const ScopedThreadPoolOverride &) = delete;

 private:
  ThreadPool *previous_;
};

}  // namespace k2

#endif  // K2_CSRC_THREAD_POOL_H_

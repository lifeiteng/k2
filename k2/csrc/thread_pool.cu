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

#include <utility>

#include "k2/csrc/thread_pool.h"

namespace k2 {

namespace {

// Caller-scope pool override (bound/restored by ScopedThreadPoolOverride).
// It only affects the thread that owns this TLS slot; task threads are
// forbidden from obtaining a pool or binding an override.
thread_local ThreadPool *t_active_pool = nullptr;

// True on threads owned by a ThreadPool that execute ProcessTasks().
thread_local bool t_is_task_thread = false;

// RAII marker for pool-owned task threads.
class TaskThreadMarker {
 public:
  TaskThreadMarker() { t_is_task_thread = true; }
  ~TaskThreadMarker() { t_is_task_thread = false; }
  TaskThreadMarker(const TaskThreadMarker &) = delete;
  TaskThreadMarker &operator=(const TaskThreadMarker &) = delete;
};

}  // namespace

static int32_t GetDefaultNumThreads() {
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 1;
  return num_threads;
}

ThreadPool::ThreadPool(int32_t num_threads)
    : threads_(num_threads > 0 ? num_threads : GetDefaultNumThreads()) {
  // Exception-safe thread startup: if a std::thread constructor throws (thread
  // limit reached, etc.), the joinable threads started so far would make the
  // destructor call std::terminate. Stop and join them, then rethrow. No task
  // can have been submitted yet, so keep_running_ = false makes them exit.
  size_t started = 0;
  try {
    for (auto &thread : threads_) {
      thread = std::thread([this]() { this->ProcessTasks(); });
      ++started;
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      keep_running_ = false;
      not_empty_cond_.notify_all();
    }
    for (size_t i = 0; i < started; ++i) threads_[i].join();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  WaitAllTasksFinished();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    keep_running_ = false;
    not_empty_cond_.notify_all();  // wake up all threads in the pool
  }
  for (auto &thread : threads_) thread.join();
}

void ThreadPool::WaitAllTasksFinished() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!finished_) {
    // wait for the `empty_cond_` condition.
    empty_cond_.wait(lock);
  }
}

bool ThreadPool::AllTasksFinished() {
  std::lock_guard<std::mutex> lock(mutex_);
  return finished_;
}

void ThreadPool::ProcessTasks() {
  // Mark pool-owned task threads so that GetThreadPool() can fail fast when a
  // task body tries to reach a pool (see the pool-scope contract in the
  // header).
  TaskThreadMarker marker;
  std::unique_lock<std::mutex> lock(mutex_);
  while (keep_running_) {
    while (tasks_.empty() && keep_running_) {
      // Wait for the `not_empty_` condition.
      // SubmitTask() or the destructor will signal it.
      not_empty_cond_.wait(lock);
    }

    if (!keep_running_) break;

    {
      auto task = std::move(tasks_.front());
      tasks_.pop();
      if (!tasks_.empty()) not_empty_cond_.notify_one();
      ++running_counter_;
      lock.unlock();  // let other threads proceed
      task();
      // any resource associated with `task` is freed here
    }

    lock.lock();
    --running_counter_;
    if (tasks_.empty() && running_counter_ == 0) {
      finished_ = true;
      // if WaitAllTasksFinished() is waiting, wake it up
      empty_cond_.notify_one();
    }

    // the lock is still held here as we assume
    // that on entering the while() loop, the lock is locked by us
  }
}

bool IsThreadPoolTaskThread() { return t_is_task_thread; }

ThreadPool *GetThreadPool() {
  // A task thread must not reach across pools: k2 tasks never call
  // GetThreadPool() (only caller-side intersect/connect code does), so this is
  // a program invariant violation. Fail fast instead of silently handing out
  // the global pool to a task that belongs to another execution scope.
  if (t_is_task_thread) {
    K2_LOG(FATAL) << "GetThreadPool() called from a thread pool task thread; "
                     "k2 tasks must not obtain a pool";
  }
  if (t_active_pool) return t_active_pool;
  static ThreadPool *pool = nullptr;
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    // TODO(fangjun): how to determine the number
    // of threads in the pool?
    //
    // It is never freed
    pool = new ThreadPool(2);
  });
  K2_CHECK_NE(pool, nullptr);
  return pool;
}

ScopedThreadPoolOverride::ScopedThreadPoolOverride(ThreadPool *pool)
    : previous_(t_active_pool) {
  K2_CHECK_NE(pool, nullptr);
  // Same invariant as GetThreadPool(): a task thread must not change its pool
  // visibility. Fail fast (K2_LOG(FATAL) throws runtime_error unless K2_ABORT
  // is set).
  if (t_is_task_thread) {
    K2_LOG(FATAL) << "ScopedThreadPoolOverride constructed on a thread pool "
                     "task thread";
  }
  t_active_pool = pool;
}

ScopedThreadPoolOverride::~ScopedThreadPoolOverride() { t_active_pool = previous_; }

}  // namespace k2

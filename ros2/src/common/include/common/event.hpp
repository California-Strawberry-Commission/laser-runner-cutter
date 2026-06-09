#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

/**
 * Concurrency primitive that provides a shared flag that can be set and waited
 * on.
 */
class Event {
 public:
  Event() : flag_(false) {}

  void set() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = true;
    cv_.notify_all();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = false;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return flag_; });
  }

  bool wait_for(float timeoutSecs) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cv_.wait_for(lock, std::chrono::duration<float>(timeoutSecs),
                        [this] { return flag_; });
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  bool flag_;
};

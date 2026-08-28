#pragma once

#include <functional>
#include <mutex>

/**
 * Thread-safe slot for a single callback of type
 * std::function<void(typename MessageT::SharedPtr)>. A node can own one
 * instance per topic and have its single, persistent subscription (created
 * once, before the executor starts spinning) forward every message through
 * it via invoke(). Whichever consumer is currently active registers its own
 * callback via set() when it starts and clears it via clear() when it stops.
 *
 * This exists so consumers never need to create or destroy ROS subscriptions of
 * their own while the executor is spinning - doing so from a thread the
 * executor doesn't own races with the executor's internal bookkeeping and can
 * crash.
 */
template <typename MessageT>
class CallbackRegistry {
 public:
  using Callback = std::function<void(typename MessageT::SharedPtr)>;

  void set(Callback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(callback);
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = nullptr;
  }

  void invoke(const typename MessageT::SharedPtr& msg) const {
    Callback callback;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      callback = callback_;
    }
    if (callback) {
      callback(msg);
    }
  }

 private:
  mutable std::mutex mutex_;
  Callback callback_;
};

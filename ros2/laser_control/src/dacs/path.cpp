#include "laser_control/dacs/path.hpp"

#include <algorithm>

Path::Path(const Point& start, const Point& end, float durationMs)
    : start_(start), end_(end), durationMs_(durationMs) {}

void Path::start() {
  if (isRunning_) {
    return;
  }

  isRunning_ = true;
  startTime_ = std::chrono::steady_clock::now();
}

void Path::reset() { isRunning_ = false; }

bool Path::isRunning() const { return isRunning_; }

Path::Point Path::getCurrentPoint() const {
  if (!isRunning_) {
    return start_;
  }

  auto now{std::chrono::steady_clock::now()};
  float elapsedMs{
      std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
          now - startTime_)
          .count()};
  float t{std::clamp(elapsedMs / durationMs_, 0.0f, 1.0f)};

  return {start_.x + (end_.x - start_.x) * t,
          start_.y + (end_.y - start_.y) * t};
}

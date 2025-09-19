#include "laser_control/dacs/path.hpp"

#include <algorithm>

Path::Path(uint32_t id, const Point& origin)
    : id_(id), origin_(origin), destination_(origin) {}

void Path::setDestination(const Point& destination, float durationMs) {
  origin_ = getCurrentPoint();
  destination_ = destination;
  durationMs_ = durationMs;
  startTime_ = std::chrono::steady_clock::now();
}

Point Path::getCurrentPoint() const {
  if (durationMs_ <= 0.0f) {
    return destination_;
  }

  auto now{std::chrono::steady_clock::now()};
  float elapsedMs{
      std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
          now - startTime_)
          .count()};
  float t{std::clamp(elapsedMs / durationMs_, 0.0f, 1.0f)};

  return {origin_.x + (destination_.x - origin_.x) * t,
          origin_.y + (destination_.y - origin_.y) * t};
}

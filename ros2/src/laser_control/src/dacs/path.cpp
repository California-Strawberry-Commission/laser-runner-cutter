#include "laser_control/dacs/path.hpp"

#include <algorithm>

namespace {

std::chrono::system_clock::time_point toTimePoint(double timestampSec) {
  return std::chrono::system_clock::time_point{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::duration<double>(timestampSec))};
}

}  // namespace

Path::Path(uint32_t id) : id_(id) {}

void Path::addWaypoint(const Point& destination, double timestampSec) {
  if (timestampSec <= 0.0) {
    setPoint(destination);
    return;
  }

  auto time{toTimePoint(timestampSec)};
  auto latestTime{
      upcoming_.empty()
          ? (last_ ? last_->time : std::chrono::system_clock::time_point::min())
          : upcoming_.back().time};
  // Ignore stale or out-of-order updates
  if (time <= latestTime) {
    return;
  }

  upcoming_.push_back({time, destination});
}

void Path::setPoint(const Point& destination) {
  upcoming_.clear();
  last_ = {std::chrono::system_clock::now(), destination};
  holdingStaticPoint_ = true;
}

std::optional<Point> Path::getCurrentPoint() {
  // Advance past any queued waypoints whose arrival time has passed, updating
  // `last_` to the most recently reached one.
  auto now{std::chrono::system_clock::now()};
  while (!upcoming_.empty() && upcoming_.front().time <= now) {
    last_ = upcoming_.front();
    upcoming_.pop_front();
    holdingStaticPoint_ = false;
  }

  if (!last_) {
    return std::nullopt;
  }

  if (holdingStaticPoint_) {
    // We are still holding the point added via `setPoint`
    return last_->point;
  }

  if (upcoming_.empty()) {
    // We have reached the end of a timed path with nothing queued, so render
    // nothing
    return std::nullopt;
  }

  // Interpolate between the last and next upcoming waypoint
  const Waypoint& next{upcoming_.front()};
  float durationSec{
      std::chrono::duration<float>(next.time - last_->time).count()};
  if (durationSec <= 0.0f) {
    return next.point;
  }

  float elapsedSec{std::chrono::duration<float>(now - last_->time).count()};
  float t{std::clamp(elapsedSec / durationSec, 0.0f, 1.0f)};

  return Point{last_->point.x + (next.point.x - last_->point.x) * t,
               last_->point.y + (next.point.y - last_->point.y) * t};
}

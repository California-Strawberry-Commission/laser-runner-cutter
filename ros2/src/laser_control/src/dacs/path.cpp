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
  auto now{std::chrono::system_clock::now()};
  auto time{timestampSec <= 0.0 ? now : toTimePoint(timestampSec)};

  if (time <= now) {
    // The arrival time has already passed, so move to `destination` immediately
    // and hold there, discarding any queued waypoints.
    upcoming_.clear();
    last_ = {now, destination};
    return;
  }

  if (upcoming_.empty() && last_) {
    // We're resuming from a held point (either the end of a prior path, or an
    // immediate arrival added above). Treat the hold as starting now, so that
    // we interpolate from the current position rather than from a stale arrival
    // time.
    last_->time = now;
  }

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

std::optional<Point> Path::getCurrentPoint() {
  // Advance past any queued waypoints whose arrival time has passed, updating
  // `last_` to the most recently reached one.
  auto now{std::chrono::system_clock::now()};
  while (!upcoming_.empty() && upcoming_.front().time <= now) {
    last_ = upcoming_.front();
    upcoming_.pop_front();
  }

  if (!last_) {
    return std::nullopt;
  }

  if (upcoming_.empty()) {
    // We have reached the end of the path with nothing queued, so keep
    // rendering the last waypoint reached until a new one comes in.
    return last_->point;
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

#pragma once

#include <chrono>
#include <deque>
#include <optional>

struct Point {
  float x;
  float y;
};

/**
 * A path traced out by a sequence of timestamped waypoints. Each call to
 * `addWaypoint` queues a new waypoint (destination + arrival time), and
 * `getCurrentPoint` interpolates between the queued waypoints based on the
 * current time, so that the traced path passes through each waypoint
 * at its associated time. Once the path reaches its last waypoint, it holds
 * there until a new waypoint is added.
 */
class Path {
 public:
  /**
   * @param id Path ID.
   */
  explicit Path(uint32_t id);

  /**
   * Queue a new destination for the path to arrive at. Stale or out-of-order
   * updates are ignored.
   *
   * @param destination Destination point, with values normalized to [0, 1].
   * (0, 0) corresponds to bottom left.
   * @param timestampSec Timestamp, in seconds since epoch, at which the
   * path should arrive at `destination`. A value <= now moves to `destination`
   * immediately, and discards any queued waypoints.
   */
  void addWaypoint(const Point& destination, double timestampSec);

  /**
   * Get the interpolated point for the current wall-clock time, based on the
   * queue of waypoints received so far.
   *
   * @return The point at the current wall-clock time, or `std::nullopt` if the
   * path hasn't reached a waypoint yet.
   */
  std::optional<Point> getCurrentPoint();

 private:
  struct Waypoint {
    std::chrono::system_clock::time_point time;
    Point point;
  };

  uint32_t id_;
  // Most recently reached waypoint, used as the interpolation origin
  std::optional<Waypoint> last_;
  // Upcoming waypoints not yet reached, sorted by ascending time
  std::deque<Waypoint> upcoming_;
};

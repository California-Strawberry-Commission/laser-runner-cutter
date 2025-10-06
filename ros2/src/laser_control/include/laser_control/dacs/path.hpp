#pragma once

#include <chrono>

struct Point {
  float x;
  float y;
};

class Path {
 public:
  /**
   * @param id Path ID.
   * @param origin Start point, with values normalized to [0, 1]. (0, 0)
   * corresponds to bottom left.
   */
  explicit Path(uint32_t id, const Point& origin);

  /**
   * Set a new destination and duration.
   *
   * @param destination Destination point, with value normalized to [0, 1]. (0,
   * 0) corresponds to bottom left.
   * @param durationMs The duration of the path in milliseconds.
   */
  void setDestination(const Point& destination, float durationMs);

  /**
   * Get the interpolated point based on the time elapsed since `start()` was
   * called.
   *
   * @return The point at the current time.
   */
  Point getCurrentPoint() const;

 private:
  uint32_t id_;
  Point origin_{0, 0};
  Point destination_{0, 0};
  float durationMs_{0.0f};
  std::chrono::steady_clock::time_point startTime_;
};
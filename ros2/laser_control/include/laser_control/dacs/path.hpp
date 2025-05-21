#pragma once

#include <chrono>

class Path {
 public:
  struct Point {
    float x;
    float y;
  };

  /**
   * @param start Start point, with values normalized to [0, 1]. (0, 0)
   * corresponds to bottom left.
   * @param end End point, with value normalized to [0, 1]. (0, 0) corresponds
   * to bottom left.
   * @param durationMs The duration of the path in milliseconds.
   */
  Path(const Point& start, const Point& end, float durationMs);

  /**
   * Begin the interpolation.
   */
  void start();

  /**
   * Reset the interpolation.
   */
  void reset();

  /**
   * Whether the path interpolation is running.
   *
   * @return Whether the path interpolation is running.
   */
  bool isRunning() const;

  /**
   * Get the interpolated point based on the time elapsed since `start()` was
   * called.
   *
   * @return The point at the current time.
   */
  Point getCurrentPoint() const;

 private:
  Point start_{0, 0};
  Point end_{0, 0};
  float durationMs_{0};
  std::chrono::steady_clock::time_point startTime_;
  bool isRunning_{false};
};
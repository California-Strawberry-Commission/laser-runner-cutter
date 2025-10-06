#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "laser_control/dacs/path.hpp"

class DAC {
 public:
  virtual ~DAC() = default;

  /**
   * Search for online DACs.
   *
   * @return number of online DACs found.
   */
  virtual int initialize() = 0;

  /**
   * Connect to the specified DAC.
   *
   * @param dacIdx Index of the DAC to connect to.
   */
  virtual void connect(int dacIdx) = 0;

  /**
   * @return whether the DAC is connected.
   */
  virtual bool isConnected() const = 0;

  /**
   * @return whether the DAC is playing.
   */
  virtual bool isPlaying() const = 0;

  /**
   * Set the color of the laser.
   *
   * @param r Red channel, with value normalized to [0, 1].
   * @param g Green channel, with value normalized to [0, 1].
   * @param b Blue channel, with value normalized to [0, 1].
   * @param i Intensity channel, with value normalized to [0, 1].
   */
  virtual void setColor(float r, float g, float b, float i) = 0;

  /**
   * Start playback of points.
   *
   * @param fps Target frames per second.
   * @param pps Target points per second. This should not exceed the capability
   * of the DAC and laser projector.
   * @param transitionDurationMs Duration in ms to turn the laser off between
   * subsequent points in the same frame. If we are rendering more than one
   * point, we need to provide enough time between subsequent points, or else
   * there may be visible streaks between the points as the galvos take time to
   * move to the new position.
   */
  virtual void play(int fps = 30, int pps = 30000,
                    float transitionDurationMs = 0.5) = 0;

  /**
   * Stop playback of points.
   */
  virtual void stop() = 0;

  /**
   * Close connection to laser DAC.
   */
  virtual void close() = 0;

  /**
   * Whether the specified path exists.
   *
   * @return Whether the specified path exists.
   */
  bool hasPath(uint32_t pathId);

  /**
   * Set the destination point of the specified path.
   *
   * @param pathId The ID of the path to set the destination.
   * @param destination The destination point.
   * @param durationMs The duration, in ms, to take from the current point to
   * the destination point.
   */
  void setPath(uint32_t pathId, const Point& destination, float durationMs);

  /**
   * Remove the specified path.
   *
   * @param pathId The ID of the path to remove.
   * @return Whether the specified path was removed.
   */
  bool removePath(uint32_t pathId);

  /**
   * Remove all paths.
   */
  void clearPaths();

 protected:
  std::unordered_map<uint32_t, std::unique_ptr<Path>> paths_;
  std::mutex pathsMutex_;
};
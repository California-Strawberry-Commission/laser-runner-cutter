#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "HeliosDac.h"
#include "laser_control_cpp/dacs/dac.hpp"

class Helios final : public DAC {
 public:
  // Helios DAC uses 12 bits (unsigned) for x and y
  static constexpr int X_MAX = 4095;
  static constexpr int Y_MAX = 4095;
  // Helios DAC uses 8 bits (unsigned) for r, g, b, i
  static constexpr int MAX_COLOR = 255;

  Helios();
  ~Helios() override;

  /**
   * Search for online DACs.
   *
   * @return number of online DACs found.
   */
  int initialize() override;

  /**
   * Connect to the specified DAC.
   *
   * @param dacIdx Index of the DAC to connect to.
   */
  void connect(int dacIdx) override;

  /**
   * @return whether the DAC is connected.
   */
  bool isConnected() const override;

  /**
   * @return whether the DAC is playing.
   */
  bool isPlaying() const override;

  /**
   * Set the color of the laser.
   *
   * @param r Red channel, with value normalized to [0, 1].
   * @param g Green channel, with value normalized to [0, 1].
   * @param b Blue channel, with value normalized to [0, 1].
   * @param i Intensity channel, with value normalized to [0, 1].
   */
  void setColor(float r, float g, float b, float i) override;

  /**
   * Add a point to be rendered by the DAC. (0, 0) corresponds to bottom left.
   * The point will be ignored if it lies outside the bounds.
   *
   * @param x x coordinate, normalized to [0, 1].
   * @param y y coordinate, normalized to [0, 1].
   */
  void addPoint(float x, float y) override;

  /**
   * Remove the last added point.
   */
  void removePoint() override;

  /**
   * Remove all points.
   */
  void clearPoints() override;

  /**
   * Start playback of points.
   * Helios max rate: 65535 pps
   * Helios max points per frame (pps/fps): 4096
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
  void play(int fps = 30, int pps = 30000,
            float transitionDurationMs = 0.5f) override;

  /**
   * Stop playback of points.
   */
  void stop() override;

  /**
   * Close connection to laser DAC.
   */
  void close() override;

 private:
  std::shared_ptr<HeliosDac> heliosDac_;
  std::atomic<bool> initialized_{false};
  int dacIdx_{-1};
  std::vector<std::pair<float, float>> points_;
  std::mutex pointsMutex_;
  std::tuple<float, float, float, float> color_;
  std::atomic<bool> playing_{false};
  std::atomic<bool> checkConnection_{false};
  std::thread checkConnectionThread_;
  std::thread playbackThread_;

  std::vector<HeliosPoint> getFrame(int fps, int pps,
                                    float transitionDurationMs);
  int getNativeStatus() const;
};
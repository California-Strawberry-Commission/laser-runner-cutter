#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "etherdream.h"
#include "laser_control/dacs/dac.hpp"

class EtherDream final : public DAC {
 public:
  // Ether Dream DAC uses 16 bits (signed) for x and y
  static constexpr int16_t X_MIN = -32768;
  static constexpr int16_t X_MAX = 32767;
  static constexpr int16_t Y_MIN = -32768;
  static constexpr int16_t Y_MAX = 32767;
  // Ether Dream DAC uses 16 bits (unsigned) for r, g, b, i
  static constexpr uint16_t MAX_COLOR = 65535;

  ~EtherDream() override;

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
   * Add a path to be rendered by the DAC.
   *
   * @param path The path to render.
   */
  void addPath(const Path& path) override;

  /**
   * Remove the last added path.
   */
  void removePath() override;

  /**
   * Remove all paths.
   */
  void clearPaths() override;

  /**
   * Start playback of points.
   * Ether Dream max rate: 100K pps
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
  std::vector<etherdream_point> getFrame(int fps, int pps,
                                         float transitionDurationMs);
  std::pair<int16_t, int16_t> denormalizePoint(float x, float y) const;
  std::tuple<uint16_t, uint16_t, uint16_t, uint16_t> denormalizeColor(
      float r, float g, float b, float i) const;
  std::string dacIdToHex(unsigned long dacId) const;

  std::atomic<bool> dacConnected_{false};
  unsigned long connectedDacId_{0};
  std::vector<Path> paths_;
  std::mutex pathsMutex_;
  std::tuple<float, float, float, float> color_;
  std::atomic<bool> playing_{false};
  std::atomic<bool> checkConnection_{false};
  std::thread checkConnectionThread_;
  std::thread playbackThread_;
};
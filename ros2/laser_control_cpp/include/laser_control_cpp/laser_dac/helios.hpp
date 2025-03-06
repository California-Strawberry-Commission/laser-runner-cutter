#pragma once

#include <dlfcn.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "laser_control_cpp/laser_dac/laser_dac.hpp"

// Helios DAC uses 12 bits (unsigned) for x and y
constexpr int X_MAX = 4095;
constexpr int Y_MAX = 4095;

// Helios DAC uses 8 bits (unsigned) for r, g, b, i
constexpr int MAX_COLOR = 255;

struct HeliosPoint {
  uint16_t x, y;
  uint8_t r, g, b, i;

  HeliosPoint(uint16_t x = 0, uint16_t y = 0, uint8_t r = 0, uint8_t g = 0,
              uint8_t b = 0, uint8_t i = 0)
      : x(x), y(y), r(r), g(g), b(b), i(i) {}
};

class HeliosDAC final : public LaserDAC {
 public:
  HeliosDAC();
  ~HeliosDAC() override;

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
  void* libHandle_{nullptr};
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
  int getNativeStatus();

  using LibOpenDevicesFunc = int (*)();
  using LibCloseDevicesFunc = void (*)();
  using LibGetStatusFunc = int (*)(int);
  using LibWriteFrameFunc = void (*)(int, int, int, HeliosPoint*, int);
  using LibStopFunc = void (*)(int);

  LibOpenDevicesFunc libOpenDevices;
  LibCloseDevicesFunc libCloseDevices;
  LibGetStatusFunc libGetStatus;
  LibWriteFrameFunc libWriteFrame;
  LibStopFunc libStop;
};
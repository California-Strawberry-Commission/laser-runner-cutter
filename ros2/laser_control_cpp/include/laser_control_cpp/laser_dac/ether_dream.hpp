#pragma once

#include <dlfcn.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "laser_control_cpp/laser_dac/laser_dac.hpp"

struct EtherDreamPoint {
  int16_t x, y;
  uint16_t r, g, b, i, u1, u2;

  EtherDreamPoint(int16_t x = 0, int16_t y = 0, uint16_t r = 0, uint16_t g = 0,
                  uint16_t b = 0, uint16_t i = 0, uint16_t u1 = 0,
                  uint16_t u2 = 0)
      : x(x), y(y), r(r), g(g), b(b), i(i), u1(u1), u2(u2) {}
};

class EtherDreamDAC final : public LaserDAC {
 public:
  // Ether Dream DAC uses 16 bits (signed) for x and y
  static constexpr int16_t X_MIN = -32768;
  static constexpr int16_t X_MAX = 32767;
  static constexpr int16_t Y_MIN = -32768;
  static constexpr int16_t Y_MAX = 32767;
  // Ether Dream DAC uses 16 bits (unsigned) for r, g, b, i
  static constexpr uint16_t MAX_COLOR = 65535;

  EtherDreamDAC();
  ~EtherDreamDAC() override;

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
  void* libHandle_{nullptr};
  std::atomic<bool> dacConnected_{false};
  unsigned long connectedDacId_{0};

  std::vector<std::pair<float, float>> points_;
  std::mutex pointsMutex_;
  std::tuple<float, float, float, float> color_;
  std::atomic<bool> playing_{false};
  std::atomic<bool> checkConnection_{false};
  std::thread checkConnectionThread_;
  std::thread playbackThread_;

  std::string dacIdToHex(unsigned long dacId);
  std::vector<EtherDreamPoint> getFrame(int fps, int pps,
                                        float transitionDurationMs);
  std::pair<int16_t, int16_t> denormalizePoint(float x, float y);
  std::tuple<uint16_t, uint16_t, uint16_t, uint16_t> denormalizeColor(float r,
                                                                      float g,
                                                                      float b,
                                                                      float i);

  using LibStartFunc = int (*)();
  using LibDacCountFunc = int (*)();
  using LibGetIdFunc = unsigned long (*)(int);
  using LibConnectFunc = int (*)(unsigned long);
  using LibIsConnectedFunc = int (*)(unsigned long);
  using LibWaitForReadyFunc = int (*)(unsigned long);
  using LibWriteFunc = int (*)(unsigned long, EtherDreamPoint*, int, int, int);
  using LibStopFunc = int (*)(unsigned long);
  using LibDisconnectFunc = void (*)(unsigned long);

  LibStartFunc libStart;
  LibDacCountFunc libDacCount;
  LibGetIdFunc libGetId;
  LibConnectFunc libConnect;
  LibIsConnectedFunc libIsConnected;
  LibWaitForReadyFunc libWaitForReady;
  LibWriteFunc libWrite;
  LibStopFunc libStop;
  LibDisconnectFunc libDisconnect;
};
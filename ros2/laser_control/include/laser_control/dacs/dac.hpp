#pragma once

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
   * Add a path to be rendered by the DAC.
   *
   * @param path The path to render.
   */
  virtual void addPath(const Path& path) = 0;

  /**
   * Remove the last added path.
   */
  virtual void removePath() = 0;

  /**
   * Remove all paths.
   */
  virtual void clearPaths() = 0;

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
};
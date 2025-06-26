#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "runner_cutter_control/tracking/track.hpp"
#include "runner_cutter_control/common_types.hpp"

/**
 * Tracker maintains a collection of Tracks with state management. Thread-safe.
 * New tracks start in the PENDING state. PENDING tracks are maintained in a
 * queue for FIFO ordering.
 */
class Tracker {
 public:
  Tracker();

  /**
   * Check if any track is in the specified state.
   *
   * @param state The state.
   * @return Whether a track in that state exists.
   */
  bool hasTrackWithState(Track::State state) const;

  /**
   * Get tracks that are in a specified state.
   *
   * @param state The desired state.
   * @return Tracks that are in that state.
   */
  std::vector<std::shared_ptr<Track>> getTracksWithState(
      Track::State state) const;

  /**
   * Get a track by ID.
   *
   * @param trackId The track ID.
   * @return The track, if it exists.
   */
  std::optional<std::shared_ptr<Track>> getTrack(uint32_t trackId) const;

  std::unordered_map<uint32_t, std::shared_ptr<Track>> getTracks() const;

  /**
   * Add a track to list of current tracks.
   *
   * @param trackId Unique instance ID assigned to the object. Must be a
   * positive integer.
   * @param pixel Pixel coordinates (x, y) of the target in the camera frame.
   * @param position 3D position (x, y, z) of the target in camera-space.
   * @param timestampMs Timestamp, in ms, of the camera frame.
   * @param confidence Confidence score associated with the detected target.
   * @return The newly created track, or existing track if it already exists.
   */
  std::shared_ptr<Track> addTrack(uint32_t trackId, const PixelCoord& pixel,
                                  const Position& position,
                                  double timestampMs, float confidence = 1.0f);

  /**
   * Get the next pending track.
   *
   * @return The next pending track.
   */
  std::optional<std::shared_ptr<Track>> getNextPendingTrack();

  /**
   * Change the state of a track.
   *
   * @param trackId The track ID.
   * @param newState The new state.
   */
  void processTrack(uint32_t trackId, Track::State newState);

  /**
   * Clear all tracks from the tracker.
   */
  void clear();

  /**
   * Get a summary of all tracks with the count of each state.
   */
  std::unordered_map<Track::State, size_t> getSummary() const;

 private:
  std::unordered_map<uint32_t, std::shared_ptr<Track>> tracks_;
  std::deque<std::shared_ptr<Track>> pendingTracks_;
  mutable std::mutex tracksMutex_;
};
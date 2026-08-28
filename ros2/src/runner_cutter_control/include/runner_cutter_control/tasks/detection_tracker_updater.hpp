#pragma once

#include <memory>
#include <unordered_set>

#include "detection_interfaces/msg/detection_result.hpp"
#include "detection_interfaces/msg/object_instance.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"

/**
 * Applies DetectionResult messages to a Tracker. Adds/updates tracks, fails
 * PENDING/ACTIVE tracks that haven't been redetected within a miss-tolerance
 * window, and requeues redetected FAILED tracks as PENDING (up to a max
 * number of attempts).
 */
class DetectionTrackerUpdater {
 public:
  /**
   * @param tracker Tracker to add/update tracks in as detections come in.
   * @param trackMissTimeoutSecs Grace period, in seconds, to tolerate a
   * PENDING or ACTIVE track not appearing in a detection frame before
   * marking it FAILED.
   * @param targetAttempts Max number of times a FAILED track may be
   * requeued as PENDING after being redetected. A negative number means no
   * limit.
   */
  DetectionTrackerUpdater(std::shared_ptr<Tracker> tracker,
                          float trackMissTimeoutSecs, int targetAttempts);
  ~DetectionTrackerUpdater() = default;

  /**
   * Process one DetectionResult message.
   *
   * @param msg The detection result to apply to the Tracker.
   * @return Whether the set of PENDING tracks changed as a result.
   */
  bool update(const detection_interfaces::msg::DetectionResult& msg);

 private:
  std::shared_ptr<Tracker> tracker_;
  float trackMissTimeoutSecs_;
  int targetAttempts_;
};

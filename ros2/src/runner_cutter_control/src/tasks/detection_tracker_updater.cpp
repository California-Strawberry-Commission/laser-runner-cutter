#include "runner_cutter_control/tasks/detection_tracker_updater.hpp"

#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "spdlog/spdlog.h"

DetectionTrackerUpdater::DetectionTrackerUpdater(
    std::shared_ptr<Tracker> tracker, float trackMissTimeoutSecs,
    int targetAttempts)
    : tracker_(std::move(tracker)),
      trackMissTimeoutSecs_(trackMissTimeoutSecs),
      targetAttempts_(targetAttempts) {}

bool DetectionTrackerUpdater::update(
    const detection_interfaces::msg::DetectionResult& msg) {
  // Newly detected tracks are added to tracker as pending. For tracks that are
  // detected again, update the track pixel and position. For failed tracks,
  // set them as pending since they may have moved since the last detection.

  std::unordered_set<uint32_t> prevPendingTracks;
  for (const auto& track :
       tracker_->getTracksWithState(Track::State::PENDING)) {
    prevPendingTracks.insert(track->getId());
  }

  std::unordered_set<uint32_t> detectedTrackIds;
  double timestampSecs{rclcpp::Time(msg.timestamp).seconds()};
  for (const auto& instance : msg.instances) {
    // A track ID of 0 is invalid (indicates that there is no track ID
    // associated with the instance)
    if (instance.track_id <= 0) {
      continue;
    }

    // Attempt to add the track to the Tracker.
    PixelCoord pixel{static_cast<int>(std::round(instance.point.x)),
                     static_cast<int>(std::round(instance.point.y))};
    Position position{static_cast<float>(instance.position.x),
                      static_cast<float>(instance.position.y),
                      static_cast<float>(instance.position.z)};

    std::shared_ptr<const Track> track;
    try {
      track = tracker_->addOrUpdateTrack(instance.track_id, pixel, position,
                                         timestampSecs, instance.confidence);
    } catch (const std::exception& e) {
      continue;
    }

    detectedTrackIds.insert(instance.track_id);

    // Put detected tracks that are marked as failed back into the pending
    // queue, since we want to reattempt to burn them (up to targetAttempts_
    // times) as they could now potentially be in bounds.
    if (track->getState() == Track::State::FAILED &&
        (targetAttempts_ < 0 ||
         track->getStateCount(Track::State::FAILED) <
             static_cast<std::size_t>(targetAttempts_))) {
      spdlog::info("Track {} was FAILED but redetected. Marking as PENDING.",
                   track->getId());
      tracker_->transitionTrackState(track->getId(), Track::State::PENDING);
    }
  }

  // Calculate estimated velocity vector
  const auto& flowDisplacement{msg.flow_displacement};
  bool flowAvailable{flowDisplacement.delta_time_secs > 0.0};
  Velocity flowVelocity{};
  if (flowAvailable) {
    flowVelocity = {
        static_cast<float>(flowDisplacement.position_displacement.x /
                           flowDisplacement.delta_time_secs),
        static_cast<float>(flowDisplacement.position_displacement.y /
                           flowDisplacement.delta_time_secs),
        static_cast<float>(flowDisplacement.position_displacement.z /
                           flowDisplacement.delta_time_secs)};
  }

  // Fail any PENDING or ACTIVE track that hasn't been detected within the
  // miss-tolerance window.
  std::vector<std::shared_ptr<const Track>> trackedTracks{
      tracker_->getTracksWithState(Track::State::PENDING)};
  auto activeTracks{tracker_->getTracksWithState(Track::State::ACTIVE)};
  trackedTracks.insert(trackedTracks.end(), activeTracks.begin(),
                       activeTracks.end());
  for (const auto& track : trackedTracks) {
    uint32_t trackId{track->getId()};
    if (detectedTrackIds.find(trackId) != detectedTrackIds.end()) {
      continue;
    }

    // This is a PENDING or ACTIVE track that has not been detected this
    // frame. Use estimated velocity derived from optical flow (if available)
    // to do a velocity-only update on the predictor
    if (flowAvailable) {
      tracker_->updateTrackVelocity(trackId, flowVelocity, timestampSecs, 0.5f);
    }

    if (timestampSecs - track->getTimestampSecs() > trackMissTimeoutSecs_) {
      spdlog::info(
          "Track {} is PENDING or ACTIVE and has not been detected within "
          "{} secs. Marking as FAILED.",
          trackId, trackMissTimeoutSecs_);
      tracker_->transitionTrackState(trackId, Track::State::FAILED);
    }
  }

  // Report whether the pending tracks have changed
  std::unordered_set<uint32_t> pendingTracks;
  for (const auto& track :
       tracker_->getTracksWithState(Track::State::PENDING)) {
    pendingTracks.insert(track->getId());
  }
  return prevPendingTracks != pendingTracks;
}

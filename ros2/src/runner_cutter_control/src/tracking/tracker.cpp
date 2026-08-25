#include "runner_cutter_control/tracking/tracker.hpp"

#include <algorithm>
#include <stdexcept>

#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

bool Tracker::hasTrackWithState(Track::State state) const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  auto it{tracksByState_.find(state)};
  return it != tracksByState_.end() && !it->second.empty();
}

std::vector<std::shared_ptr<const Track>> Tracker::getTracksWithState(
    Track::State state) const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  std::vector<std::shared_ptr<const Track>> result;
  auto stateIt{tracksByState_.find(state)};
  if (stateIt == tracksByState_.end()) {
    return result;
  }
  result.reserve(stateIt->second.size());
  for (uint32_t id : stateIt->second) {
    result.push_back(tracks_.at(id));
  }
  return result;
}

std::optional<std::shared_ptr<const Track>> Tracker::getTrack(
    uint32_t trackId) const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  auto it{tracks_.find(trackId)};
  if (it != tracks_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::unordered_map<uint32_t, std::shared_ptr<const Track>> Tracker::getTracks()
    const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  std::unordered_map<uint32_t, std::shared_ptr<const Track>> result;
  result.reserve(tracks_.size());
  for (const auto& [id, track] : tracks_) {
    result[id] = track;
  }
  return result;
}

std::shared_ptr<const Track> Tracker::addOrUpdateTrack(uint32_t trackId,
                                                       const PixelCoord& pixel,
                                                       const Position& position,
                                                       double timestampSecs,
                                                       float confidence) {
  if (trackId == 0) {
    throw std::invalid_argument("Track ID must be positive");
  }

  std::lock_guard<std::mutex> lock(tracksMutex_);
  std::shared_ptr<Track> track;
  if (tracks_.find(trackId) != tracks_.end()) {
    // If the track exists, update it
    track = tracks_[trackId];
    track->setPixel(pixel);
    track->setPosition(position);
    track->setTimestampSecs(timestampSecs);
  } else {
    // Create a new track and set as PENDING
    track = std::make_shared<Track>(trackId, pixel, position, timestampSecs,
                                    Track::State::PENDING,
                                    std::make_unique<KalmanFilterPredictor>());
    tracks_[trackId] = track;
    pendingTracks_.push_back(trackId);
    tracksByState_[Track::State::PENDING].insert(trackId);
  }

  // Add position measurement to the Track's predictor
  track->getPredictor().add(timestampSecs, position, confidence);

  return track;
}

std::optional<std::shared_ptr<const Track>>
Tracker::activateNextPendingTrack() {
  std::lock_guard<std::mutex> lock(tracksMutex_);

  if (pendingTracks_.empty()) {
    return std::nullopt;
  }

  uint32_t nextTrackId{pendingTracks_.front()};
  pendingTracks_.pop_front();
  auto nextTrack{tracks_.at(nextTrackId)};
  tracksByState_[Track::State::PENDING].erase(nextTrackId);
  nextTrack->setState(Track::State::ACTIVE);
  tracksByState_[Track::State::ACTIVE].insert(nextTrackId);
  return nextTrack;
}

bool Tracker::transitionTrackState(uint32_t trackId, Track::State newState) {
  std::lock_guard<std::mutex> lock(tracksMutex_);

  auto it{tracks_.find(trackId)};
  if (it == tracks_.end()) {
    return false;
  }

  auto& track{it->second};
  Track::State oldState{track->getState()};
  if (oldState == newState) {
    return false;
  }

  // If the track is leaving the PENDING state, remove it from pendingTracks_
  if (oldState == Track::State::PENDING) {
    auto pendingIt{
        std::find(pendingTracks_.begin(), pendingTracks_.end(), trackId)};
    if (pendingIt != pendingTracks_.end()) {
      pendingTracks_.erase(pendingIt);
    }
  }

  tracksByState_[oldState].erase(trackId);
  track->setState(newState);
  tracksByState_[newState].insert(trackId);

  // If the track is entering the PENDING state, add it to pendingTracks_
  if (newState == Track::State::PENDING) {
    pendingTracks_.push_back(trackId);
  }

  // Reset the predictor if the track is COMPLETED or FAILED
  if (newState == Track::State::COMPLETED || newState == Track::State::FAILED) {
    track->getPredictor().reset();
  }

  return true;
}

bool Tracker::updateTrackVelocity(uint32_t trackId, const Velocity& velocity,
                                  double timestampSecs, float confidence) {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  auto it{tracks_.find(trackId)};
  if (it == tracks_.end()) {
    return false;
  }

  return it->second->getPredictor().addVelocity(timestampSecs, velocity,
                                                confidence);
}

void Tracker::clear() {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  tracks_.clear();
  pendingTracks_.clear();
  tracksByState_.clear();
}

std::unordered_map<Track::State, size_t> Tracker::getCountsByState() const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  std::unordered_map<Track::State, size_t> countsByState;
  for (const auto& [state, ids] : tracksByState_) {
    if (!ids.empty()) {
      countsByState[state] = ids.size();
    }
  }
  return countsByState;
}

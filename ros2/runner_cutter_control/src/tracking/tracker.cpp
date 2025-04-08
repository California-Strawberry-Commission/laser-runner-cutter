#include "runner_cutter_control/tracking/tracker.hpp"

#include <algorithm>
#include <stdexcept>

#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

Tracker::Tracker() {}

bool Tracker::hasTrackWithState(Track::State state) const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  return std::any_of(
      tracks_.begin(), tracks_.end(),
      [state](const auto& track) { return track.second->getState() == state; });
}

std::vector<std::shared_ptr<Track>> Tracker::getTracksWithState(
    Track::State state) const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  std::vector<std::shared_ptr<Track>> result;
  for (const auto& track : tracks_) {
    if (track.second->getState() == state) {
      result.push_back(track.second);
    }
  }
  return result;
}

std::optional<std::shared_ptr<Track>> Tracker::getTrack(int trackId) const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  auto it{tracks_.find(trackId)};
  if (it != tracks_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::unordered_map<int, std::shared_ptr<Track>> Tracker::getTracks() const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  // Return a copy for thread-safety
  return tracks_;
}

std::shared_ptr<Track> Tracker::addTrack(
    int trackId, std::pair<int, int> pixel,
    std::tuple<float, float, float> position, double timestampMs,
    float confidence) {
  if (trackId <= 0) {
    throw std::invalid_argument("Track ID must be positive");
  }

  std::lock_guard<std::mutex> lock(tracksMutex_);
  std::shared_ptr<Track> track;
  if (tracks_.find(trackId) != tracks_.end()) {
    // If the track exists, update it
    track = tracks_[trackId];
    track->setPixel(pixel);
    track->setPosition(position);
  } else {
    // Create a new track and set as PENDING
    track =
        std::make_shared<Track>(trackId, pixel, position, Track::State::PENDING,
                                std::make_unique<KalmanFilterPredictor>());
    tracks_[trackId] = track;
    pendingTracks_.push_back(track);
  }

  // Update predictor for the track
  track->getPredictor().add(position, timestampMs, confidence);

  return track;
}

std::optional<std::shared_ptr<Track>> Tracker::getNextPendingTrack() {
  std::lock_guard<std::mutex> lock(tracksMutex_);

  if (pendingTracks_.empty()) {
    return std::nullopt;
  }

  auto nextTrack{pendingTracks_.front()};
  pendingTracks_.pop_front();
  nextTrack->setState(Track::State::ACTIVE);
  return nextTrack;
}

void Tracker::processTrack(int trackId, Track::State newState) {
  auto trackOpt{getTrack(trackId)};
  if (!trackOpt) {
    return;
  }

  auto track{trackOpt.value()};
  if (track->getState() == newState) {
    return;
  }

  std::lock_guard<std::mutex> lock(tracksMutex_);

  // If the track is leaving the PENDING state, remove it from pendingTracks_
  if (track->getState() == Track::State::PENDING) {
    auto it{std::find(pendingTracks_.begin(), pendingTracks_.end(), track)};
    if (it != pendingTracks_.end()) {
      pendingTracks_.erase(it);
    }
  }

  track->setState(newState);

  // If the track is entering the PENDING state, add it to pendingTracks_
  if (newState == Track::State::PENDING) {
    pendingTracks_.push_back(track);
  }
}

void Tracker::clear() {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  tracks_.clear();
  pendingTracks_.clear();
}

std::unordered_map<Track::State, int> Tracker::getSummary() const {
  std::lock_guard<std::mutex> lock(tracksMutex_);
  std::unordered_map<Track::State, int> summary;
  for (const auto& track : tracks_) {
    summary[track.second->getState()]++;
  }
  return summary;
}
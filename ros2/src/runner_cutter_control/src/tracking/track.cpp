#include "runner_cutter_control/tracking/track.hpp"

Track::Track(uint32_t id, const PixelCoord& pixel, const Position& position,
             double timestampSecs, Track::State state,
             std::unique_ptr<Predictor> predictor)
    : id_{id},
      pixel_{pixel},
      position_{position},
      timestampSecs_{timestampSecs},
      predictor_{predictor ? std::move(predictor) : nullptr} {
  stateCount_ = {{Track::State::PENDING, 0},
                 {Track::State::ACTIVE, 0},
                 {Track::State::COMPLETED, 0},
                 {Track::State::FAILED, 0}};
  setState(state);
}

PixelCoord Track::getPixel() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pixel_;
}

Position Track::getPosition() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return position_;
}

double Track::getTimestampSecs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return timestampSecs_;
}

Track::State Track::getState() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

size_t Track::getStateCount(Track::State state) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = stateCount_.find(state);
  if (it != stateCount_.end()) {
    return it->second;
  } else {
    return 0;
  }
}

void Track::setPixel(const PixelCoord& pixel) {
  std::lock_guard<std::mutex> lock(mutex_);
  pixel_ = pixel;
}

void Track::setPosition(const Position& position) {
  std::lock_guard<std::mutex> lock(mutex_);
  position_ = position;
}

void Track::setState(Track::State state) {
  std::lock_guard<std::mutex> lock(mutex_);
  state_ = state;
  stateCount_[state]++;
}

void Track::setTimestampSecs(double timestampSecs) {
  std::lock_guard<std::mutex> lock(mutex_);
  timestampSecs_ = timestampSecs;
}
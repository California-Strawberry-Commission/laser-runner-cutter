#include "runner_cutter_control_cpp/tracking/track.hpp"

Track::Track(int id, std::pair<int, int> pixel,
             std::tuple<float, float, float> position, Track::State state,
             std::unique_ptr<Predictor> predictor)
    : id_{id},
      pixel_{pixel},
      position_{position},
      predictor_(predictor ? std::move(predictor) : nullptr) {
  stateCount_ = {{Track::State::PENDING, 0},
                 {Track::State::ACTIVE, 0},
                 {Track::State::COMPLETED, 0},
                 {Track::State::FAILED, 0}};
  setState(state);
}

int Track::getStateCount(Track::State state) const {
  auto it = stateCount_.find(state);
  if (it != stateCount_.end()) {
    return it->second;
  } else {
    return 0;
  }
}

void Track::setPixel(std::pair<int, int> pixel) { pixel_ = pixel; }

void Track::setPosition(std::tuple<float, float, float> position) {
  position_ = position;
}

void Track::setState(Track::State state) {
  state_ = state;
  stateCount_[state]++;
}

#pragma once

#include <memory>
#include <unordered_map>

#include "runner_cutter_control/prediction/predictor.hpp"

class Track {
 public:
  enum class State {
    PENDING,    // Still needs to be burned
    ACTIVE,     // Actively in the process of being targeted and burned
    COMPLETED,  // Has successfully been burned
    FAILED      // Failed to burn
  };

  explicit Track(uint32_t id, std::pair<int, int> pixel,
                 std::tuple<float, float, float> position,
                 State state = State::PENDING,
                 std::unique_ptr<Predictor> predictor = nullptr);

  uint32_t getId() const { return id_; };
  std::pair<int, int> getPixel() const { return pixel_; };
  std::tuple<float, float, float> getPosition() const { return position_; };
  State getState() const { return state_; };
  size_t getStateCount(State state) const;
  Predictor& getPredictor() { return *predictor_; }
  const Predictor& getPredictor() const { return *predictor_; }

  void setPixel(std::pair<int, int> pixel);
  void setPosition(std::tuple<float, float, float> position);
  void setState(State state);

 private:
  uint32_t id_{0};
  std::pair<int, int> pixel_;
  std::tuple<float, float, float> position_;
  std::unordered_map<State, size_t> stateCount_;
  State state_{State::PENDING};
  std::unique_ptr<Predictor> predictor_;
};
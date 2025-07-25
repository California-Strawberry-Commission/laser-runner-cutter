#pragma once

#include <memory>
#include <unordered_map>

#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/prediction/predictor.hpp"

class Track {
 public:
  enum class State {
    PENDING,    // Still needs to be burned
    ACTIVE,     // Actively in the process of being targeted and burned
    COMPLETED,  // Has successfully been burned
    FAILED      // Failed to burn
  };

  explicit Track(uint32_t id, const PixelCoord& pixel, const Position& position,
                 State state = State::PENDING,
                 std::unique_ptr<Predictor> predictor = nullptr);

  uint32_t getId() const { return id_; };
  PixelCoord getPixel() const { return pixel_; };
  Position getPosition() const { return position_; };
  State getState() const { return state_; };
  size_t getStateCount(State state) const;
  Predictor& getPredictor() { return *predictor_; }
  const Predictor& getPredictor() const { return *predictor_; }

  void setPixel(const PixelCoord& pixel);
  void setPosition(const Position& position);
  void setState(State state);

 private:
  uint32_t id_{0};
  PixelCoord pixel_;
  Position position_;
  std::unordered_map<State, size_t> stateCount_;
  State state_{State::PENDING};
  std::unique_ptr<Predictor> predictor_;
};
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
                 double timestampMs, State state = State::PENDING,
                 std::unique_ptr<Predictor> predictor = nullptr);
  ~Track() = default;

  uint32_t getId() const { return id_; };
  PixelCoord getPixel() const { return pixel_; };
  Position getPosition() const { return position_; };
  double getTimestampMs() const { return timestampMs_; };
  State getState() const { return state_; };
  size_t getStateCount(State state) const;
  Predictor& getPredictor() { return *predictor_; }
  const Predictor& getPredictor() const { return *predictor_; }

  void setPixel(const PixelCoord& pixel);
  void setPosition(const Position& position);
  void setTimestampMs(double timestampMs);
  void setState(State state);

 private:
  uint32_t id_{0};
  PixelCoord pixel_;
  Position position_;
  double timestampMs_;
  std::unordered_map<State, size_t> stateCount_;
  State state_{State::PENDING};
  std::unique_ptr<Predictor> predictor_;
};
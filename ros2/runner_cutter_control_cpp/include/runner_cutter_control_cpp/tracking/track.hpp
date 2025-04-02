#pragma once

#include <unordered_map>

class Track {
 public:
  enum class State {
    PENDING,    // Still needs to be burned
    ACTIVE,     // Actively in the process of being targeted and burned
    COMPLETED,  // Has successfully been burned
    FAILED      // Failed to burn
  };

  Track(int id, std::pair<int, int> pixel,
        std::tuple<float, float, float> position, State state = State::PENDING);

  int getId() const { return id_; };
  std::pair<int, int> getPixel() const { return pixel_; };
  std::tuple<float, float, float> getPosition() const { return position_; };
  State getState() const { return state_; };
  int getStateCount(State state) const;

  void setPixel(std::pair<int, int> pixel);
  void setPosition(std::tuple<float, float, float> position);
  void setState(State state);

 private:
  int id_{0};
  std::pair<int, int> pixel_;
  std::tuple<float, float, float> position_;
  std::unordered_map<State, int> stateCount_;
  State state_{State::PENDING};
};
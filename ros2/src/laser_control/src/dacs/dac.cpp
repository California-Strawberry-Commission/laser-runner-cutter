#include "laser_control/dacs/dac.hpp"

bool DAC::hasPath(uint32_t pathId) {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  return paths_.find(pathId) != paths_.end();
}

void DAC::setPath(uint32_t pathId, const Point& destination, float durationMs) {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  auto it = paths_.find(pathId);
  if (it != paths_.end()) {
    // If path already exists, update the destination
    it->second->setDestination(destination, durationMs);
  } else {
    // If path does not exist, create it
    paths_.emplace(pathId, std::make_unique<Path>(pathId, destination));
  }
}

bool DAC::removePath(uint32_t pathId) {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  return paths_.erase(pathId) > 0;
}

void DAC::clearPaths() {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  paths_.clear();
}
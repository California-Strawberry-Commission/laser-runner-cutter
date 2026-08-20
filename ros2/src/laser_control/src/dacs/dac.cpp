#include "laser_control/dacs/dac.hpp"

bool DAC::hasPath(uint32_t pathId) {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  return paths_.find(pathId) != paths_.end();
}

void DAC::addWaypoint(uint32_t pathId, const Point& destination,
                      double timestampSec) {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  auto it = paths_.find(pathId);
  if (it == paths_.end()) {
    it = paths_.emplace(pathId, std::make_unique<Path>(pathId)).first;
  }
  it->second->addWaypoint(destination, timestampSec);
}

void DAC::setPoint(uint32_t pathId, const Point& destination) {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  auto it = paths_.find(pathId);
  if (it == paths_.end()) {
    it = paths_.emplace(pathId, std::make_unique<Path>(pathId)).first;
  }
  it->second->setPoint(destination);
}

bool DAC::removePath(uint32_t pathId) {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  return paths_.erase(pathId) > 0;
}

void DAC::clearPaths() {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  paths_.clear();
}

bool DAC::hasPaths() {
  std::lock_guard<std::mutex> lock(pathsMutex_);
  return !paths_.empty();
}

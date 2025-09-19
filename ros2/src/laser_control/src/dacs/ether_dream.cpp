#include "laser_control/dacs/ether_dream.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

EtherDream::~EtherDream() { close(); }

int EtherDream::initialize() {
  spdlog::info("Initializing Ether Dream DAC");
  etherdream_lib_start();
  spdlog::info("Finding available Ether Dream DACs...");

  // Ether Dream DACs broadcast once per second, so we need to wait for a bit
  // longer than that to ensure that we see broadcasts from all online DACs
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));

  int dacCount{etherdream_dac_count()};
  spdlog::info("Found {} Ether Dream DACs.", dacCount);
  return dacCount;
}

void EtherDream::connect(int dacIdx) {
  spdlog::info("Connecting to DAC...");
  unsigned long dacId{etherdream_get_id(dacIdx)};
  if (etherdream_connect(dacId) < 0) {
    throw std::runtime_error("Could not connect to DAC [" + dacIdToHex(dacId) +
                             "]");
  }
  connectedDacId_ = dacId;
  dacConnected_ = true;
  spdlog::info("Connected to DAC with ID: {}", dacIdToHex(connectedDacId_));

  auto checkConnectionFunc{[this]() {
    while (checkConnection_) {
      if (etherdream_is_connected(connectedDacId_) == 0) {
        spdlog::warn("DAC connection error. Attempting to reconnect.");
      }
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  }};

  if (!checkConnectionThread_.joinable()) {
    checkConnection_ = true;
    checkConnectionThread_ = std::thread(checkConnectionFunc);
  }
}

bool EtherDream::isConnected() const {
  return dacConnected_ && etherdream_is_connected(connectedDacId_) > 0;
}

bool EtherDream::isPlaying() const { return playing_; }

void EtherDream::setColor(float r, float g, float b, float i) {
  color_ = {r, g, b, i};
}

void EtherDream::play(int fps, int pps, float transitionDurationMs) {
  if (playing_) {
    return;
  }

  fps = std::max(0, fps);
  // Ether Dream max rate: 100K pps
  pps = std::clamp(pps, 0, 100000);
  playing_ = true;

  playbackThread_ = std::thread([this, fps, pps, transitionDurationMs]() {
    while (playing_) {
      std::vector<etherdream_point> frame{
          getFrame(fps, pps, transitionDurationMs)};

      etherdream_wait_for_ready(connectedDacId_);
      etherdream_write(connectedDacId_, frame.data(), frame.size(),
                       frame.size() * fps, 1);
    }
    etherdream_stop(connectedDacId_);
  });
}

void EtherDream::stop() {
  if (!playing_) {
    return;
  }

  playing_ = false;
  if (playbackThread_.joinable()) {
    playbackThread_.join();
  }
}

void EtherDream::close() {
  stop();

  if (checkConnection_) {
    checkConnection_ = false;
    if (checkConnectionThread_.joinable()) {
      checkConnectionThread_.join();
    }
  }

  etherdream_stop(connectedDacId_);
  dacConnected_ = false;
  etherdream_disconnect(connectedDacId_);
}

std::vector<etherdream_point> EtherDream::getFrame(int fps, int pps,
                                                   float transitionDurationMs) {
  // We'll use "laxel", or laser "pixel", to refer to each point that the laser
  // projector renders, which disambiguates it from "point", which refers to the
  // (x, y) coordinates we want to have rendered

  std::lock_guard<std::mutex> lock(pathsMutex_);

  // Calculate how many laxels of transition we need to add per point
  int laxelsPerTransition{
      static_cast<int>(std::round(transitionDurationMs / (1000.0f / pps)))};

  // Calculate how many laxels we render each point
  float ppf{static_cast<float>(pps) / fps};
  int numPoints{static_cast<int>(paths_.size())};
  int laxelsPerPoint{static_cast<int>(
      (numPoints == 0) ? std::round(ppf) : std::round(ppf / numPoints))};
  int laxelsPerFrame{(numPoints == 0) ? laxelsPerPoint
                                      : laxelsPerPoint * numPoints};

  // Prepare frame
  std::vector<etherdream_point> frame(laxelsPerFrame);

  // Extract color components from tuple and convert to DAC range
  auto [rNorm, gNorm, bNorm, iNorm]{color_};
  auto [r, g, b, i]{denormalizeColor(rNorm, gNorm, bNorm, iNorm)};

  if (numPoints == 0) {
    // Even if there are no points to render, we still to send over laxels so
    // that we don't underflow the DAC buffer
    for (int laxelIdx = 0; laxelIdx < laxelsPerFrame; ++laxelIdx) {
      frame[laxelIdx] = {0, 0, 0, 0, 0, 0, 0, 0};
    }
  } else {
    int pathIdx{0};
    for (const auto& [pathId, path] : paths_) {
      auto point{path->getCurrentPoint()};

      for (int laxelIdx = 0; laxelIdx < laxelsPerPoint; ++laxelIdx) {
        // Pad BEFORE the "on" laxel so that the galvo settles first, and only
        // if there is more than one point
        bool isTransition{numPoints > 1 && laxelIdx < laxelsPerTransition};
        int frameLaxelIdx{static_cast<int>(pathIdx) * laxelsPerPoint +
                          laxelIdx};

        auto pointDenorm = denormalizePoint(point.x, point.y);
        frame[frameLaxelIdx] = {pointDenorm.first,
                                pointDenorm.second,
                                isTransition ? static_cast<uint16_t>(0) : r,
                                isTransition ? static_cast<uint16_t>(0) : g,
                                isTransition ? static_cast<uint16_t>(0) : b,
                                isTransition ? static_cast<uint16_t>(0) : i,
                                0,
                                0};
      }
    }

    ++pathIdx;
  }

  return frame;
}

std::pair<int16_t, int16_t> EtherDream::denormalizePoint(float x,
                                                         float y) const {
  int16_t xDenorm{static_cast<int16_t>(std::round(
      (EtherDream::X_MAX - EtherDream::X_MIN) * x + EtherDream::X_MIN))};
  int16_t yDenorm{static_cast<int16_t>(std::round(
      (EtherDream::Y_MAX - EtherDream::Y_MIN) * y + EtherDream::Y_MIN))};
  return {xDenorm, yDenorm};
}

std::tuple<uint16_t, uint16_t, uint16_t, uint16_t> EtherDream::denormalizeColor(
    float r, float g, float b, float i) const {
  return {static_cast<uint16_t>(std::round(r * EtherDream::MAX_COLOR)),
          static_cast<uint16_t>(std::round(g * EtherDream::MAX_COLOR)),
          static_cast<uint16_t>(std::round(b * EtherDream::MAX_COLOR)),
          static_cast<uint16_t>(std::round(i * EtherDream::MAX_COLOR))};
}

std::string EtherDream::dacIdToHex(unsigned long dacId) const {
  std::stringstream ss;
  ss << "0x" << std::hex << dacId;
  return ss.str();
}

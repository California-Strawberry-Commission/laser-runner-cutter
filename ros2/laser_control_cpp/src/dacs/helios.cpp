#include "laser_control_cpp/dacs/helios.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>

Helios::~Helios() { close(); }

int Helios::initialize() {
  int num_devices{heliosDac_.OpenDevices()};
  initialized_ = true;
  spdlog::info("Found {} Helios DACs.", num_devices);
  return num_devices;
}

void Helios::connect(int dacIdx) {
  if (!initialized_) {
    return;
  }

  dacIdx_ = dacIdx;

  auto checkConnectionFunc{[this]() {
    while (checkConnection_) {
      if (getNativeStatus() < 0) {
        spdlog::warn("DAC error {}. Attempting to reconnect.",
                     getNativeStatus());
        stop();
        if (initialized_) {
          heliosDac_.CloseDevices();
          initialized_ = false;
        }
        initialize();
      }
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  }};

  if (!checkConnectionThread_.joinable()) {
    checkConnection_ = true;
    checkConnectionThread_ = std::thread(checkConnectionFunc);
  }
}

bool Helios::isConnected() const {
  return dacIdx_ >= 0 && getNativeStatus() >= 0;
}

bool Helios::isPlaying() const { return playing_; }

void Helios::setColor(float r, float g, float b, float i) {
  color_ = {r, g, b, i};
}

void Helios::addPoint(float x, float y) {
  if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f) {
    std::lock_guard<std::mutex> lock(pointsMutex_);
    points_.emplace_back(x, y);
  }
}

void Helios::removePoint() {
  std::lock_guard<std::mutex> lock(pointsMutex_);
  if (!points_.empty()) {
    points_.pop_back();
  }
}

void Helios::clearPoints() {
  std::lock_guard<std::mutex> lock(pointsMutex_);
  points_.clear();
}

void Helios::play(int fps, int pps, float transitionDurationMs) {
  if (!initialized_ || playing_) {
    return;
  }

  fps = std::max(0, fps);
  // Helios max rate: 65535 pps
  pps = std::clamp(pps, 0, 65535);
  playing_ = true;

  playbackThread_ = std::thread([this, fps, pps, transitionDurationMs]() {
    while (playing_) {
      std::vector<HeliosPoint> frame{getFrame(fps, pps, transitionDurationMs)};

      int statusAttempts = 0;
      while (statusAttempts < 512 && getNativeStatus() != 1) {
        statusAttempts++;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }

      if (!frame.empty()) {
        heliosDac_.WriteFrame(dacIdx_, frame.size() * fps, 0, frame.data(),
                              frame.size());
      }
    }
    heliosDac_.Stop(dacIdx_);
  });
}

void Helios::stop() {
  if (!playing_) {
    return;
  }

  playing_ = false;
  if (playbackThread_.joinable()) {
    playbackThread_.join();
  }
}

void Helios::close() {
  stop();

  if (checkConnection_) {
    checkConnection_ = false;
    if (checkConnectionThread_.joinable()) {
      checkConnectionThread_.join();
    }
  }

  if (initialized_) {
    heliosDac_.CloseDevices();
    initialized_ = false;
  }
  dacIdx_ = -1;
}

std::vector<HeliosPoint> Helios::getFrame(int fps, int pps,
                                          float transitionDurationMs) {
  // We'll use "laxel", or laser "pixel", to refer to each point that the laser
  // projector renders, which disambiguates it from "point", which refers to the
  // (x, y) coordinates we want to have rendered

  std::lock_guard<std::mutex> lock(pointsMutex_);

  // Calculate how many laxels of transition we need to add per point
  int laxelsPerTransition{
      static_cast<int>(std::round(transitionDurationMs / (1000.0f / pps)))};

  // Calculate how many laxels we render each point
  float ppf{static_cast<float>(pps) / fps};
  int numPoints{static_cast<int>(points_.size())};
  int laxelsPerPoint{static_cast<int>(
      (numPoints == 0) ? std::round(ppf) : std::round(ppf / numPoints))};
  int laxelsPerFrame{(numPoints == 0) ? laxelsPerPoint
                                      : laxelsPerPoint * numPoints};

  // Prepare frame
  std::vector<HeliosPoint> frame(laxelsPerFrame);

  // Extract color components from tuple and convert to DAC range
  auto [rNorm, gNorm, bNorm, iNorm]{color_};
  uint8_t r{static_cast<uint8_t>(std::round(rNorm * Helios::MAX_COLOR))};
  uint8_t g{static_cast<uint8_t>(std::round(gNorm * Helios::MAX_COLOR))};
  uint8_t b{static_cast<uint8_t>(std::round(bNorm * Helios::MAX_COLOR))};
  uint8_t i{static_cast<uint8_t>(std::round(iNorm * Helios::MAX_COLOR))};

  if (numPoints == 0) {
    // Even if there are no points to render, we still to send over laxels so
    // that we don't underflow the DAC buffer
    for (int laxelIdx = 0; laxelIdx < laxelsPerFrame; ++laxelIdx) {
      frame[laxelIdx] = {0, 0, 0, 0, 0, 0};
    }
  } else {
    for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx) {
      auto [x, y]{points_[pointIdx]};
      for (int laxelIdx = 0; laxelIdx < laxelsPerPoint; ++laxelIdx) {
        // Pad BEFORE the "on" laxel so that the galvo settles first, and only
        // if there is more than one point
        bool isTransition{numPoints > 1 && laxelIdx < laxelsPerTransition};
        int frameLaxelIdx{static_cast<int>(pointIdx) * laxelsPerPoint +
                          laxelIdx};

        frame[frameLaxelIdx] = {
            static_cast<uint16_t>(
                std::round(x * Helios::X_MAX)),  // convert to DAC range
            static_cast<uint16_t>(
                std::round(y * Helios::Y_MAX)),  // convert to DAC range
            isTransition ? static_cast<uint8_t>(0) : r,
            isTransition ? static_cast<uint8_t>(0) : g,
            isTransition ? static_cast<uint8_t>(0) : b,
            isTransition ? static_cast<uint8_t>(0) : i};
      }
    }
  }

  return frame;
}

int Helios::getNativeStatus() const {
  // 1 means ready to receive frame
  // 0 means not ready to receive frame
  // Any negative status means error
  return const_cast<HeliosDac&>(heliosDac_).GetStatus(dacIdx_);
}
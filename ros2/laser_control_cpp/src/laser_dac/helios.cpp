#include "laser_control_cpp/laser_dac/helios.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>

HeliosDAC::HeliosDAC() {
  libHandle_ = dlopen("libHeliosDacAPI.so", RTLD_LAZY);
  if (!libHandle_) {
    spdlog::error("Failed to load library: {}", dlerror());
    exit(1);
  }

  libOpenDevices = (LibOpenDevicesFunc)dlsym(libHandle_, "OpenDevices");
  libCloseDevices = (LibCloseDevicesFunc)dlsym(libHandle_, "CloseDevices");
  libGetStatus = (LibGetStatusFunc)dlsym(libHandle_, "GetStatus");
  libWriteFrame = (LibWriteFrameFunc)dlsym(libHandle_, "WriteFrame");
  libStop = (LibStopFunc)dlsym(libHandle_, "Stop");
}

HeliosDAC::~HeliosDAC() {
  close();
  dlclose(libHandle_);
}

int HeliosDAC::initialize() {
  int num_devices{libOpenDevices()};
  spdlog::info("Found {} Helios DACs.", num_devices);
  return num_devices;
}

void HeliosDAC::connect(int dacIdx) {
  dacIdx_ = dacIdx;

  auto checkConnectionFunc{[this]() {
    while (checkConnection_) {
      if (getNativeStatus() < 0) {
        spdlog::warn("DAC error {}. Attempting to reconnect.",
                     getNativeStatus());
        stop();
        libCloseDevices();
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

bool HeliosDAC::isConnected() const {
  return dacIdx_ >= 0 && libGetStatus(dacIdx_) >= 0;
}

bool HeliosDAC::isPlaying() const { return playing_; }

void HeliosDAC::setColor(float r, float g, float b, float i) {
  color_ = {r, g, b, i};
}

void HeliosDAC::addPoint(float x, float y) {
  if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f) {
    std::lock_guard<std::mutex> lock(pointsMutex_);
    points_.emplace_back(x, y);
  }
}

void HeliosDAC::removePoint() {
  std::lock_guard<std::mutex> lock(pointsMutex_);
  if (!points_.empty()) {
    points_.pop_back();
  }
}

void HeliosDAC::clearPoints() {
  std::lock_guard<std::mutex> lock(pointsMutex_);
  points_.clear();
}

void HeliosDAC::play(int fps, int pps, float transitionDurationMs) {
  if (playing_) {
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
        libWriteFrame(dacIdx_, frame.size() * fps, 0, frame.data(),
                      frame.size());
      }
    }
    libStop(dacIdx_);
  });
}

void HeliosDAC::stop() {
  if (!playing_) {
    return;
  }

  playing_ = false;
  if (playbackThread_.joinable()) {
    playbackThread_.join();
  }
}

void HeliosDAC::close() {
  stop();

  if (checkConnection_) {
    checkConnection_ = false;
    if (checkConnectionThread_.joinable()) {
      checkConnectionThread_.join();
    }
  }

  libCloseDevices();
  dacIdx_ = -1;
}

std::vector<HeliosPoint> HeliosDAC::getFrame(int fps, int pps,
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
  float r_f, g_f, b_f, i_f;
  std::tie(r_f, g_f, b_f, i_f) = color_;
  int r{static_cast<int>(std::round(r_f * HeliosDAC::MAX_COLOR))};
  int g{static_cast<int>(std::round(g_f * HeliosDAC::MAX_COLOR))};
  int b{static_cast<int>(std::round(b_f * HeliosDAC::MAX_COLOR))};
  int i{static_cast<int>(std::round(i_f * HeliosDAC::MAX_COLOR))};

  if (numPoints == 0) {
    // Even if there are no points to render, we still to send over laxels so
    // that we don't underflow the DAC buffer
    for (int laxelIdx = 0; laxelIdx < laxelsPerFrame; ++laxelIdx) {
      frame[laxelIdx] = HeliosPoint(0, 0, 0, 0, 0, 0);
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

        frame[frameLaxelIdx] = HeliosPoint(
            std::round(x * HeliosDAC::X_MAX),  // convert to DAC range
            std::round(y * HeliosDAC::Y_MAX),  // convert to DAC range
            isTransition ? 0 : r, isTransition ? 0 : g, isTransition ? 0 : b,
            isTransition ? 0 : i);
      }
    }
  }

  return frame;
}

int HeliosDAC::getNativeStatus() {
  // 1 means ready to receive frame
  // 0 means not ready to receive frame
  // Any negative status means error
  return libGetStatus(dacIdx_);
}
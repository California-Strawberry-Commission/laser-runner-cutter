#include "laser_control_cpp/laser_dac/ether_dream.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

EtherDreamDAC::EtherDreamDAC() {
  libHandle_ = dlopen("libEtherDream.so", RTLD_LAZY);
  if (!libHandle_) {
    spdlog::error("Failed to load library: {}", dlerror());
    exit(1);
  }

  libStart = (LibStartFunc)dlsym(libHandle_, "etherdream_lib_start");
  libDacCount = (LibDacCountFunc)dlsym(libHandle_, "etherdream_dac_count");
  libGetId = (LibGetIdFunc)dlsym(libHandle_, "etherdream_get_id");
  libConnect = (LibConnectFunc)dlsym(libHandle_, "etherdream_connect");
  libIsConnected =
      (LibIsConnectedFunc)dlsym(libHandle_, "etherdream_is_connected");
  libWaitForReady =
      (LibWaitForReadyFunc)dlsym(libHandle_, "etherdream_wait_for_ready");
  libWrite = (LibWriteFunc)dlsym(libHandle_, "etherdream_write");
  libStop = (LibStopFunc)dlsym(libHandle_, "etherdream_stop");
  libDisconnect = (LibDisconnectFunc)dlsym(libHandle_, "etherdream_disconnect");
}

EtherDreamDAC::~EtherDreamDAC() {
  close();
  dlclose(libHandle_);
}

int EtherDreamDAC::initialize() {
  spdlog::info("Initializing Ether Dream DAC");
  libStart();
  spdlog::info("Finding available Ether Dream DACs...");

  // Ether Dream DACs broadcast once per second, so we need to wait for a bit
  // longer than that to ensure that we see broadcasts from all online DACs
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));

  int dacCount{libDacCount()};
  spdlog::info("Found {} Ether Dream DACs.", dacCount);
  return dacCount;
}

void EtherDreamDAC::connect(int dacIdx) {
  spdlog::info("Connecting to DAC...");
  unsigned long dacId = libGetId(dacIdx);
  if (libConnect(dacId) < 0) {
    throw std::runtime_error("Could not connect to DAC [" + dacIdToHex(dacId) +
                             "]");
  }
  connectedDacId_ = dacId;
  dacConnected_ = true;
  spdlog::info("Connected to DAC with ID: {}", dacIdToHex(connectedDacId_));

  auto checkConnectionFunc{[this]() {
    while (checkConnection_) {
      if (libIsConnected(connectedDacId_) == 0) {
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

bool EtherDreamDAC::isConnected() const {
  return dacConnected_ && libIsConnected(connectedDacId_) > 0;
}

bool EtherDreamDAC::isPlaying() const { return playing_; }

void EtherDreamDAC::setColor(float r, float g, float b, float i) {
  color_ = {r, g, b, i};
}

void EtherDreamDAC::addPoint(float x, float y) {
  if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f) {
    std::lock_guard<std::mutex> lock(pointsMutex_);
    points_.emplace_back(x, y);
  }
}

void EtherDreamDAC::removePoint() {
  std::lock_guard<std::mutex> lock(pointsMutex_);
  if (!points_.empty()) {
    points_.pop_back();
  }
}

void EtherDreamDAC::clearPoints() {
  std::lock_guard<std::mutex> lock(pointsMutex_);
  points_.clear();
}

void EtherDreamDAC::play(int fps, int pps, float transitionDurationMs) {
  if (playing_) {
    return;
  }

  fps = std::max(0, fps);
  // Ether Dream max rate: 100K pps
  pps = std::clamp(pps, 0, 100000);
  playing_ = true;

  playbackThread_ = std::thread([this, fps, pps, transitionDurationMs]() {
    while (playing_) {
      std::vector<EtherDreamPoint> frame{
          getFrame(fps, pps, transitionDurationMs)};

      libWaitForReady(connectedDacId_);
      libWrite(connectedDacId_, frame.data(), frame.size(), frame.size() * fps,
               1);
    }
    libStop(connectedDacId_);
  });
}

void EtherDreamDAC::stop() {
  if (!playing_) {
    return;
  }

  playing_ = false;
  if (playbackThread_.joinable()) {
    playbackThread_.join();
  }
}

void EtherDreamDAC::close() {
  stop();

  if (checkConnection_) {
    checkConnection_ = false;
    if (checkConnectionThread_.joinable()) {
      checkConnectionThread_.join();
    }
  }

  libStop(connectedDacId_);
  dacConnected_ = false;
  libDisconnect(connectedDacId_);
}

std::vector<EtherDreamPoint> EtherDreamDAC::getFrame(
    int fps, int pps, float transitionDurationMs) {
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
  std::vector<EtherDreamPoint> frame(laxelsPerFrame);

  // Extract color components from tuple and convert to DAC range
  float r_f, g_f, b_f, i_f;
  std::tie(r_f, g_f, b_f, i_f) = color_;
  auto colorDenorm = denormalizeColor(r_f, g_f, b_f, i_f);

  if (numPoints == 0) {
    // Even if there are no points to render, we still to send over laxels so
    // that we don't underflow the DAC buffer
    for (int laxelIdx = 0; laxelIdx < laxelsPerFrame; ++laxelIdx) {
      frame[laxelIdx] = EtherDreamPoint(0, 0, 0, 0, 0, 0, 0, 0);
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

        auto pointDenorm = denormalizePoint(x, y);
        frame[frameLaxelIdx] =
            EtherDreamPoint(pointDenorm.first, pointDenorm.second,
                            isTransition ? 0 : std::get<0>(colorDenorm),
                            isTransition ? 0 : std::get<1>(colorDenorm),
                            isTransition ? 0 : std::get<2>(colorDenorm),
                            isTransition ? 0 : std::get<3>(colorDenorm));
      }
    }
  }

  return frame;
}

std::pair<int16_t, int16_t> EtherDreamDAC::denormalizePoint(float x, float y) {
  int16_t xDenorm = static_cast<int16_t>(
      std::round((EtherDreamDAC::X_MAX - EtherDreamDAC::X_MIN) * x +
                 EtherDreamDAC::X_MIN));
  int16_t yDenorm = static_cast<int16_t>(
      std::round((EtherDreamDAC::Y_MAX - EtherDreamDAC::Y_MIN) * y +
                 EtherDreamDAC::Y_MIN));
  return {xDenorm, yDenorm};
}

std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>
EtherDreamDAC::denormalizeColor(float r, float g, float b, float i) {
  return {static_cast<uint16_t>(std::round(r * EtherDreamDAC::MAX_COLOR)),
          static_cast<uint16_t>(std::round(g * EtherDreamDAC::MAX_COLOR)),
          static_cast<uint16_t>(std::round(b * EtherDreamDAC::MAX_COLOR)),
          static_cast<uint16_t>(std::round(i * EtherDreamDAC::MAX_COLOR))};
}

std::string EtherDreamDAC::dacIdToHex(unsigned long dacId) {
  std::stringstream ss;
  ss << "0x" << std::hex << dacId;
  return ss.str();
}

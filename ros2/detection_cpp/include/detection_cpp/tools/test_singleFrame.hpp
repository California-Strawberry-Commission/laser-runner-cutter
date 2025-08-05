#pragma once

// #include <opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <optional>
#include <vector>
#include <filesystem>
#include <fstream>

#include "spdlog/spdlog.h"
#include "NvInfer.h"


class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    // suppress info-level messages
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} logger;
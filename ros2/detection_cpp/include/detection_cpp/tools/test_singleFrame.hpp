#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>
#include <filesystem>
#include <fstream>


#include "spdlog/spdlog.h"
#include "NvInfer.h"

#define DEFAULT_INPUT_IMAGE_WIDTH 1024
#define DEFAULT_INPUT_IMAGE_HEIGHT 768
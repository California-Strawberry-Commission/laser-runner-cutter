#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

// Convert a millisecond timestamp to a (seconds, nanoseconds) pair suitable
// for builtin_interfaces::msg::Time.
inline std::pair<int32_t, uint32_t> millisecondsToRosTime(double milliseconds) {
  int32_t seconds = static_cast<int32_t>(milliseconds / 1000.0);
  uint32_t nanoseconds =
      static_cast<uint32_t>((milliseconds - seconds * 1000.0) * 1e6);
  return {seconds, nanoseconds};
}

inline std::string expandUser(const std::string& path) {
  if (path.empty() || path[0] != '~') {
    return path;
  }

  const char* home{std::getenv("HOME")};
  if (!home) {
    throw std::runtime_error("HOME environment variable not set");
  }

  return std::string(home) + path.substr(1);
}

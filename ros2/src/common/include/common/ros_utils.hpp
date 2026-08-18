#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <rcl_interfaces/msg/log.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <string>

namespace common {

inline std::string formatRosTimestamp(
    const builtin_interfaces::msg::Time& stamp) {
  rclcpp::Time rosTime(stamp);
  auto sec{static_cast<time_t>(rosTime.seconds())};
  auto nsec{rosTime.nanoseconds() % 1'000'000'000};

  // Format date + time
  std::tm tm;
  localtime_r(&sec, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d%H%M%S");

  // Add milliseconds
  oss << std::setw(3) << std::setfill('0') << (nsec / 1'000'000);

  return oss.str();
}

inline builtin_interfaces::msg::Time nowAsRosTime() {
  auto nowNs{std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count()};
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nowNs / 1'000'000'000);
  stamp.nanosec = static_cast<uint32_t>(nowNs % 1'000'000'000);
  return stamp;
}

inline builtin_interfaces::msg::Time toRosTime(double timestampSecs) {
  double secsFloor{std::floor(timestampSecs)};
  double frac{timestampSecs - secsFloor};

  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(secsFloor);
  stamp.nanosec = static_cast<uint32_t>(std::round(frac * 1e9));

  // Guard against frac getting rounded up to exactly 1e9
  if (stamp.nanosec >= 1000000000u) {
    stamp.sec += 1;
    stamp.nanosec -= 1000000000u;
  }

  return stamp;
}

inline void publishNotification(
    const rclcpp::Logger& logger,
    const rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr& publisher,
    const std::string& msg,
    rclcpp::Logger::Level level = rclcpp::Logger::Level::Info) {
  uint8_t logMsgLevel{0};
  switch (level) {
    case rclcpp::Logger::Level::Debug:
      RCLCPP_DEBUG(logger, "%s", msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::DEBUG;
      break;
    case rclcpp::Logger::Level::Info:
      RCLCPP_INFO(logger, "%s", msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::INFO;
      break;
    case rclcpp::Logger::Level::Warn:
      RCLCPP_WARN(logger, "%s", msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::WARN;
      break;
    case rclcpp::Logger::Level::Error:
      RCLCPP_ERROR(logger, "%s", msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::ERROR;
      break;
    case rclcpp::Logger::Level::Fatal:
      RCLCPP_FATAL(logger, "%s", msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::FATAL;
      break;
    default:
      RCLCPP_ERROR(logger, "Unknown log level: %s", msg.c_str());
      return;
  }

  auto logMsg{rcl_interfaces::msg::Log()};
  logMsg.stamp = rclcpp::Clock().now();
  logMsg.level = logMsgLevel;
  logMsg.msg = msg;
  publisher->publish(std::move(logMsg));
}

}  // namespace common

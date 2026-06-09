#pragma once

#include <rcl_interfaces/msg/log.hpp>
#include <rclcpp/rclcpp.hpp>

namespace common {

inline void publishNotification(
    const rclcpp::Logger& logger,
    const rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr& publisher,
    const std::string& msg,
    rclcpp::Logger::Level level = rclcpp::Logger::Level::Info) {
  uint8_t logMsgLevel{0};
  switch (level) {
    case rclcpp::Logger::Level::Debug:
      RCLCPP_DEBUG(logger, msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::DEBUG;
      break;
    case rclcpp::Logger::Level::Info:
      RCLCPP_INFO(logger, msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::INFO;
      break;
    case rclcpp::Logger::Level::Warn:
      RCLCPP_WARN(logger, msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::WARN;
      break;
    case rclcpp::Logger::Level::Error:
      RCLCPP_ERROR(logger, msg.c_str());
      logMsgLevel = rcl_interfaces::msg::Log::ERROR;
      break;
    case rclcpp::Logger::Level::Fatal:
      RCLCPP_FATAL(logger, msg.c_str());
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

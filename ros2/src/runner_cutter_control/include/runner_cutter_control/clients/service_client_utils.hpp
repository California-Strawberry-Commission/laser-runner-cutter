#pragma once

#include <chrono>
#include <future>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace client_utils {

/**
 * Send a service request and wait up to timeoutSecs for the response.
 *
 * @param client The service client to send the request through.
 * @param request The request to send.
 * @param timeoutSecs How long, in seconds, to wait for the response.
 * @param logger Logger to report a timeout on.
 * @return The response, or nullptr if the call timed out.
 */
template <typename ServiceT>
typename ServiceT::Response::SharedPtr callService(
    const typename rclcpp::Client<ServiceT>::SharedPtr& client,
    const typename ServiceT::Request::SharedPtr& request, int timeoutSecs,
    const rclcpp::Logger& logger) {
  auto future{client->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(logger, "Service call to '%s' timed out.",
                 client->get_service_name());
    return nullptr;
  }

  return future.get();
}

/**
 * Set one or more parameters on a remote node and wait up to timeoutSecs for
 * the result.
 *
 * @param parametersClient The parameters client to set the parameters
 * through.
 * @param parameters The parameters to set.
 * @param timeoutSecs How long, in seconds, to wait for the result.
 * @param logger Logger to report a timeout or failure on.
 * @return Whether all parameters were set successfully.
 */
inline bool setParameters(
    const std::shared_ptr<rclcpp::AsyncParametersClient>& parametersClient,
    const std::vector<rclcpp::Parameter>& parameters, int timeoutSecs,
    const rclcpp::Logger& logger) {
  auto future{parametersClient->set_parameters(parameters)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(logger, "Set parameter(s) timed out.");
    return false;
  }

  for (const auto& result : future.get()) {
    if (!result.successful) {
      RCLCPP_ERROR(logger, "Failed to set parameter: %s",
                   result.reason.c_str());
      return false;
    }
  }

  return true;
}

}  // namespace client_utils

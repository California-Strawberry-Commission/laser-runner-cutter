#include "laser_control_cpp/dacs/dac.hpp"
#include "laser_control_cpp/dacs/ether_dream.hpp"
#include "laser_control_cpp/dacs/helios.hpp"
#include "laser_control_interfaces/msg/device_state.hpp"
#include "laser_control_interfaces/msg/state.hpp"
#include "laser_control_interfaces/srv/add_point.hpp"
#include "laser_control_interfaces/srv/get_state.hpp"
#include "laser_control_interfaces/srv/set_color.hpp"
#include "laser_control_interfaces/srv/set_playback_params.hpp"
#include "laser_control_interfaces/srv/set_points.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

class LaserControlNode : public rclcpp::Node {
 public:
  explicit LaserControlNode() : Node("laser_control_node") {
    /////////////
    // Parameters
    /////////////
    declare_parameter<std::string>("laser_control_params.dac_type",
                                   "helios");  // "helios" or "ether_dream"
    declare_parameter<int>("laser_control_params.dac_index", 0);
    declare_parameter<int>("laser_control_params.fps", 30);
    declare_parameter<int>("laser_control_params.pps", 30000);
    declare_parameter<float>("laser_control_params.transition_duration_ms",
                             0.5f);

    /////////
    // Topics
    /////////
    rclcpp::QoS latchedQos{rclcpp::KeepLast(1)};
    latchedQos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    statePublisher_ = create_publisher<laser_control_interfaces::msg::State>(
        "~/state", latchedQos);

    ///////////
    // Services
    ///////////
    serviceCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    startDeviceService_ = create_service<std_srvs::srv::Trigger>(
        "~/start_device",
        std::bind(&LaserControlNode::onStartDevice, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    closeDeviceService_ = create_service<std_srvs::srv::Trigger>(
        "~/close_device",
        std::bind(&LaserControlNode::onCloseDevice, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    setColorService_ = create_service<laser_control_interfaces::srv::SetColor>(
        "~/set_color",
        std::bind(&LaserControlNode::onSetColor, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    addPointService_ = create_service<laser_control_interfaces::srv::AddPoint>(
        "~/add_point",
        std::bind(&LaserControlNode::onAddPoint, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    setPointsService_ =
        create_service<laser_control_interfaces::srv::SetPoints>(
            "~/set_points",
            std::bind(&LaserControlNode::onSetPoints, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    removePointService_ = create_service<std_srvs::srv::Trigger>(
        "~/remove_point",
        std::bind(&LaserControlNode::onRemovePoint, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    clearPointsService_ = create_service<std_srvs::srv::Trigger>(
        "~/clear_points",
        std::bind(&LaserControlNode::onClearPoints, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    setPlaybackParamsService_ =
        create_service<laser_control_interfaces::srv::SetPlaybackParams>(
            "~/set_playback_params",
            std::bind(&LaserControlNode::onSetPlaybackParams, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    playService_ = create_service<std_srvs::srv::Trigger>(
        "~/play",
        std::bind(&LaserControlNode::onPlay, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    stopService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop",
        std::bind(&LaserControlNode::onStop, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    getStateService_ = create_service<laser_control_interfaces::srv::GetState>(
        "~/get_state",
        std::bind(&LaserControlNode::onGetState, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);

    ////////////
    // DAC Setup
    ////////////
    auto dacType{getParamDacType()};
    if (dacType == "helios") {
      dac_ = std::make_shared<Helios>();
    } else if (dacType == "ether_dream") {
      dac_ = std::make_shared<EtherDream>();
    } else {
      throw std::runtime_error("Unknown dac_type: " + dacType);
    }
  }

 private:
  void onStartDevice(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (dac_->isConnected()) {
      response->success = false;
      return;
    }

    connecting_ = true;
    publishState();

    int numDacs{dac_->initialize()};
    if (numDacs == 0) {
      response->success = false;
    } else {
      dac_->connect(getParamDacIndex());
      response->success = true;
    }

    connecting_ = false;
    publishState();
  }

  void onCloseDevice(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!dac_->isConnected()) {
      response->success = false;
      return;
    }

    dac_->close();
    publishState();

    response->success = true;
  }

  void onSetColor(
      const std::shared_ptr<laser_control_interfaces::srv::SetColor::Request>
          request,
      std::shared_ptr<laser_control_interfaces::srv::SetColor::Response>
          response) {
    dac_->setColor(request->r, request->g, request->b, request->i);
    response->success = true;
  }

  void onAddPoint(
      const std::shared_ptr<laser_control_interfaces::srv::AddPoint::Request>
          request,
      std::shared_ptr<laser_control_interfaces::srv::AddPoint::Response>
          response) {
    dac_->addPoint(request->point.x, request->point.y);
    response->success = true;
  }

  void onSetPoints(
      const std::shared_ptr<laser_control_interfaces::srv::SetPoints::Request>
          request,
      std::shared_ptr<laser_control_interfaces::srv::SetPoints::Response>
          response) {
    dac_->clearPoints();
    for (const auto& point : request->points) {
      dac_->addPoint(point.x, point.y);
    }
    response->success = true;
  }

  void onRemovePoint(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    dac_->removePoint();
    response->success = true;
  }

  void onClearPoints(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    dac_->clearPoints();
    response->success = true;
  }

  void onSetPlaybackParams(
      const std::shared_ptr<
          laser_control_interfaces::srv::SetPlaybackParams::Request>
          request,
      std::shared_ptr<
          laser_control_interfaces::srv::SetPlaybackParams::Response>
          response) {
    std::vector<rclcpp::Parameter> newParams{
        rclcpp::Parameter{"laser_control_params.fps",
                          static_cast<int>(request->fps)},
        rclcpp::Parameter{"laser_control_params.pps",
                          static_cast<int>(request->pps)},
        rclcpp::Parameter{"laser_control_params.transition_duration_ms",
                          static_cast<double>(request->transition_duration_ms)},
    };
    set_parameters(newParams);
    response->success = true;
  }

  void onPlay(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
              std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    dac_->play(getParamFps(), getParamPps(), getParamTransitionDurationMs());
    publishState();
    response->success = true;
  }

  void onStop(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
              std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    dac_->stop();
    publishState();
    response->success = true;
  }

  void onGetState(
      const std::shared_ptr<laser_control_interfaces::srv::GetState::Request>,
      std::shared_ptr<laser_control_interfaces::srv::GetState::Response>
          response) {
    response->state = *getState();
  }

  uint8_t getDeviceState() {
    if (connecting_) {
      return laser_control_interfaces::msg::DeviceState::CONNECTING;
    } else if (!dac_->isConnected()) {
      return laser_control_interfaces::msg::DeviceState::DISCONNECTED;
    } else if (dac_->isPlaying()) {
      return laser_control_interfaces::msg::DeviceState::PLAYING;
    } else {
      return laser_control_interfaces::msg::DeviceState::STOPPED;
    }
  }

  laser_control_interfaces::msg::State::SharedPtr getState() {
    auto msg{std::make_shared<laser_control_interfaces::msg::State>()};
    msg->device_state = getDeviceState();
    return msg;
  }

  void publishState() { statePublisher_->publish(*getState()); }

#pragma region Param helpers

  std::string getParamDacType() {
    return get_parameter("laser_control_params.dac_type").as_string();
  }

  int getParamDacIndex() {
    return static_cast<int>(
        get_parameter("laser_control_params.dac_index").as_int());
  }

  int getParamFps() {
    return static_cast<int>(get_parameter("laser_control_params.fps").as_int());
  }

  int getParamPps() {
    return static_cast<int>(get_parameter("laser_control_params.pps").as_int());
  }

  float getParamTransitionDurationMs() {
    return static_cast<float>(
        get_parameter("laser_control_params.transition_duration_ms")
            .as_double());
  }

#pragma endregion

  rclcpp::Publisher<laser_control_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr startDeviceService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr closeDeviceService_;
  rclcpp::Service<laser_control_interfaces::srv::SetColor>::SharedPtr
      setColorService_;
  rclcpp::Service<laser_control_interfaces::srv::AddPoint>::SharedPtr
      addPointService_;
  rclcpp::Service<laser_control_interfaces::srv::SetPoints>::SharedPtr
      setPointsService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr removePointService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clearPointsService_;
  rclcpp::Service<laser_control_interfaces::srv::SetPlaybackParams>::SharedPtr
      setPlaybackParamsService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr playService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopService_;
  rclcpp::Service<laser_control_interfaces::srv::GetState>::SharedPtr
      getStateService_;

  std::shared_ptr<DAC> dac_;
  bool connecting_{false};
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  // MultiThreadedExecutor allows callbacks to run in parallel
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node{std::make_shared<LaserControlNode>()};
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();

  return 0;
}
#include "laser_control/dacs/dac.hpp"
#include "laser_control/dacs/ether_dream.hpp"
#include "laser_control/dacs/helios.hpp"
#include "laser_control_interfaces/msg/device_state.hpp"
#include "laser_control_interfaces/msg/path.hpp"
#include "laser_control_interfaces/msg/state.hpp"
#include "laser_control_interfaces/srv/get_state.hpp"
#include "laser_control_interfaces/srv/set_color.hpp"
#include "laser_control_interfaces/srv/set_playback_params.hpp"
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

    /////////////
    // Publishers
    /////////////
    rclcpp::QoS latchedQos{rclcpp::KeepLast(1)};
    latchedQos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    statePublisher_ = create_publisher<laser_control_interfaces::msg::State>(
        "~/state", latchedQos);

    //////////////
    // Subscribers
    //////////////
    rclcpp::SubscriptionOptions options;
    subscriberCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    options.callback_group = subscriberCallbackGroup_;
    pathSubscriber_ = create_subscription<laser_control_interfaces::msg::Path>(
        "~/path", 1,
        std::bind(&LaserControlNode::onPath, this, std::placeholders::_1),
        options);

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

    // Publish initial state
    publishState();
  }

 private:
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

#pragma region State publishing

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

  laser_control_interfaces::msg::State::UniquePtr getStateMsg() {
    auto msg{std::make_unique<laser_control_interfaces::msg::State>()};
    msg->device_state = getDeviceState();
    return msg;
  }

  void publishState() { statePublisher_->publish(std::move(getStateMsg())); }

#pragma endregion

#pragma region Callbacks

  void onPath(const laser_control_interfaces::msg::Path::SharedPtr msg) {
    dac_->clearPaths();
    if (msg->laser_on) {
      Path::Point start{static_cast<float>(msg->start.x),
                        static_cast<float>(msg->start.y)};
      Path::Point end{static_cast<float>(msg->end.x),
                      static_cast<float>(msg->end.y)};
      Path path{start, end, msg->duration_ms};
      dac_->addPath(path);
    }
  }

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
    response->state = std::move(*getStateMsg());
  }

#pragma endregion

  rclcpp::Publisher<laser_control_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::CallbackGroup::SharedPtr subscriberCallbackGroup_;
  rclcpp::Subscription<laser_control_interfaces::msg::Path>::SharedPtr
      pathSubscriber_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr startDeviceService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr closeDeviceService_;
  rclcpp::Service<laser_control_interfaces::srv::SetColor>::SharedPtr
      setColorService_;
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
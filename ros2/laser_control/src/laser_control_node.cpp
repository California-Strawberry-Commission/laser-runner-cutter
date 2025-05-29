#include "laser_control/dacs/dac.hpp"
#include "laser_control/dacs/ether_dream.hpp"
#include "laser_control/dacs/helios.hpp"
#include "laser_control_interfaces/msg/device_state.hpp"
#include "laser_control_interfaces/msg/path.hpp"
#include "laser_control_interfaces/msg/state.hpp"
#include "laser_control_interfaces/srv/get_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

class LaserControlNode : public rclcpp::Node {
 public:
  explicit LaserControlNode() : Node("laser_control_node") {
    /////////////
    // Parameters
    /////////////
    declare_parameter<std::string>("dac_type",
                                   "helios");  // "helios" or "ether_dream"
    declare_parameter<int>("dac_index", 0);
    declare_parameter<int>("fps", 30);
    declare_parameter<int>("pps", 30000);
    declare_parameter<float>("transition_duration_ms", 0.5f);
    declare_parameter<std::vector<float>>("color", {0.15f, 0.0f, 0.0f, 0.0f});
    paramSubscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    colorParamCallback_ = paramSubscriber_->add_parameter_callback(
        "color", std::bind(&LaserControlNode::onColorChanged, this,
                           std::placeholders::_1));

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
    auto [r, g, b, i]{getParamColor()};
    dac_->setColor(r, g, b, i);

    // Publish initial state
    publishState();
  }

 private:
#pragma region Param helpers

  std::string getParamDacType() {
    return get_parameter("dac_type").as_string();
  }

  int getParamDacIndex() {
    return static_cast<int>(get_parameter("dac_index").as_int());
  }

  int getParamFps() { return static_cast<int>(get_parameter("fps").as_int()); }

  int getParamPps() { return static_cast<int>(get_parameter("pps").as_int()); }

  float getParamTransitionDurationMs() {
    return static_cast<float>(
        get_parameter("transition_duration_ms").as_double());
  }

  std::tuple<float, float, float, float> getParamColor() {
    auto param{get_parameter("color").as_double_array()};
    return {param[0], param[1], param[2], param[3]};
  }

  void onColorChanged(const rclcpp::Parameter& param) {
    std::vector<double> values{param.as_double_array()};
    if (values.size() == 4) {
      float r{static_cast<float>(values[0])};
      float g{static_cast<float>(values[1])};
      float b{static_cast<float>(values[2])};
      float i{static_cast<float>(values[3])};
      dac_->setColor(r, g, b, i);
    } else {
      RCLCPP_WARN(get_logger(), "Expected 4 values for 'color', got %zu",
                  values.size());
    }
  }

#pragma endregion

#pragma region State publishing

  uint8_t getDeviceState() {
    if (connecting_) {
      return laser_control_interfaces::msg::DeviceState::CONNECTING;
    } else if (disconnecting_) {
      return laser_control_interfaces::msg::DeviceState::DISCONNECTING;
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

    disconnecting_ = true;
    publishState();

    dac_->close();
    response->success = true;

    disconnecting_ = false;
    publishState();
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

  std::shared_ptr<rclcpp::ParameterEventHandler> paramSubscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> colorParamCallback_;
  rclcpp::Publisher<laser_control_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::CallbackGroup::SharedPtr subscriberCallbackGroup_;
  rclcpp::Subscription<laser_control_interfaces::msg::Path>::SharedPtr
      pathSubscriber_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr startDeviceService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr closeDeviceService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr playService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopService_;
  rclcpp::Service<laser_control_interfaces::srv::GetState>::SharedPtr
      getStateService_;

  std::shared_ptr<DAC> dac_;
  bool connecting_{false};
  bool disconnecting_{false};
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
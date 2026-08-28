#include <fmt/core.h>

#include <functional>

#include "camera_control_interfaces/msg/device_state.hpp"
#include "camera_control_interfaces/msg/state.hpp"
#include "common/ros_utils.hpp"
#include "common/utils.hpp"
#include "detection_interfaces/msg/detection_result.hpp"
#include "laser_control_interfaces/msg/device_state.hpp"
#include "laser_control_interfaces/msg/state.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/prediction/average_velocity_predictor.hpp"
#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"
#include "runner_cutter_control/prediction/last_known_predictor.hpp"
#include "runner_cutter_control/tasks/add_calibration_points_task.hpp"
#include "runner_cutter_control/tasks/calibration_task.hpp"
#include "runner_cutter_control/tasks/callback_registry.hpp"
#include "runner_cutter_control/tasks/circle_follower_task.hpp"
#include "runner_cutter_control/tasks/manual_target_laser_task.hpp"
#include "runner_cutter_control/tasks/runner_cutter_task.hpp"
#include "runner_cutter_control/tasks/static_runner_cutter_task.hpp"
#include "runner_cutter_control/tools/prediction_evaluator.hpp"
#include "runner_cutter_control_interfaces/msg/state.hpp"
#include "runner_cutter_control_interfaces/msg/tracks.hpp"
#include "runner_cutter_control_interfaces/srv/add_calibration_points.hpp"
#include "runner_cutter_control_interfaces/srv/calibrate.hpp"
#include "runner_cutter_control_interfaces/srv/get_state.hpp"
#include "runner_cutter_control_interfaces/srv/manual_target_laser.hpp"
#include "std_srvs/srv/trigger.hpp"

class RunnerCutterControlNode : public rclcpp::Node {
 public:
  explicit RunnerCutterControlNode() : Node("runner_cutter_control_node") {
    /////////////
    // Parameters
    /////////////
    declare_parameter<std::string>("laser_control_node_name", "laser0");
    declare_parameter<std::string>("camera_control_node_name", "camera0");
    declare_parameter<std::string>("detection_node_name", "detection0");
    declare_parameter<std::vector<int>>("calibration_grid_size", {11, 11});
    declare_parameter<std::vector<float>>("calibration_x_bounds", {0.0f, 1.0f});
    declare_parameter<std::vector<float>>("calibration_y_bounds", {0.0f, 1.0f});
    declare_parameter<std::vector<float>>("tracking_laser_color",
                                          {0.15f, 0.0f, 0.0f, 0.0f});
    declare_parameter<std::vector<float>>("burn_laser_color",
                                          {0.0f, 0.0f, 1.0f, 0.0f});
    declare_parameter<float>("burn_time_secs", 1.0);
    // Max number of times to attempt to target a detected runner to burn. An
    // attempt may fail if the runner burn point is outside the laser bounds, if
    // the aiming process failed, or if the runner was no longer detected. A
    // negative number means no limit.
    declare_parameter<int>("target_attempts", -1);
    // Duration, in seconds, during which if no viable target becomes available,
    // the runner cutter task will stop. A negative number means no auto disarm.
    declare_parameter<float>("auto_disarm_secs", -1.0);
    // Grace period, in seconds, to tolerate a track not appearing in a
    // detection frame before treating it as missing.
    declare_parameter<float>("track_miss_timeout_secs", 0.2);
    declare_parameter<std::string>("save_dir", "~/runner_cutter");
    // Whether to enable the new runner cutter task (burn-while-moving using
    // prediction)
    declare_parameter<bool>("enable_runner_cutter_v2", false);
    // How far ahead, in seconds, to predict the target's position when placing
    // laser waypoints.
    declare_parameter<float>("lookahead_secs", 0.2);

    /////////////
    // Publishers
    /////////////
    rclcpp::QoS latchedQos{rclcpp::KeepLast(1)};
    latchedQos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    statePublisher_ =
        create_publisher<runner_cutter_control_interfaces::msg::State>(
            "~/state", latchedQos);
    notificationsPublisher_ =
        create_publisher<rcl_interfaces::msg::Log>("/notifications", 1);
    tracksPublisher_ =
        create_publisher<runner_cutter_control_interfaces::msg::Tracks>(
            "~/tracks", 1);

    //////////////
    // Subscribers
    //////////////
    rclcpp::SubscriptionOptions options;
    subscriberCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    options.callback_group = subscriberCallbackGroup_;
    auto laserStateTopicName{
        fmt::format("/{}/state", getParamLaserControlNodeName())};
    laserStateSubscriber_ =
        create_subscription<laser_control_interfaces::msg::State>(
            laserStateTopicName, latchedQos,
            std::bind(&RunnerCutterControlNode::onLaserState, this,
                      std::placeholders::_1),
            options);
    auto cameraStateTopicName{
        fmt::format("/{}/state", getParamCameraControlNodeName())};
    cameraStateSubscriber_ =
        create_subscription<camera_control_interfaces::msg::State>(
            cameraStateTopicName, latchedQos,
            std::bind(&RunnerCutterControlNode::onCameraState, this,
                      std::placeholders::_1),
            options);

    // For detections, we subscribe once here at the node level and allow
    // runtime registration of callbacks inside tasks via CallbackRegistry. We
    // need to do this as there is a risk of a race condition if we attempt to
    // create/destroy a subscription from a thread that the executor doesn't
    // own.
    detectionCallbackRegistry_ = std::make_shared<
        CallbackRegistry<detection_interfaces::msg::DetectionResult>>();
    detectionsCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions detectionsOptions;
    detectionsOptions.callback_group = detectionsCallbackGroup_;
    auto detectionsTopicName{
        fmt::format("/{}/detections", getParamDetectionNodeName())};
    detectionsSubscriber_ =
        create_subscription<detection_interfaces::msg::DetectionResult>(
            detectionsTopicName, rclcpp::SensorDataQoS(),
            [this](detection_interfaces::msg::DetectionResult::SharedPtr msg) {
              detectionCallbackRegistry_->invoke(msg);
            },
            detectionsOptions);

    ///////////
    // Services
    ///////////
    serviceCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    // TODO: use action instead once there's a new release of roslib. Currently
    // roslib does not support actions with ROS2
    calibrateService_ =
        create_service<runner_cutter_control_interfaces::srv::Calibrate>(
            "~/calibrate",
            std::bind(&RunnerCutterControlNode::onCalibrate, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    saveCalibrationService_ = create_service<std_srvs::srv::Trigger>(
        "~/save_calibration",
        std::bind(&RunnerCutterControlNode::onSaveCalibration, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    loadCalibrationService_ = create_service<std_srvs::srv::Trigger>(
        "~/load_calibration",
        std::bind(&RunnerCutterControlNode::onLoadCalibration, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    addCalibrationPointsService_ = create_service<
        runner_cutter_control_interfaces::srv::AddCalibrationPoints>(
        "~/add_calibration_points",
        std::bind(&RunnerCutterControlNode::onAddCalibrationPoints, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    manualTargetLaserService_ = create_service<
        runner_cutter_control_interfaces::srv::ManualTargetLaser>(
        "~/manual_target_laser",
        std::bind(&RunnerCutterControlNode::onManualTargetLaser, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    startRunnerCutterService_ = create_service<std_srvs::srv::Trigger>(
        "~/start_runner_cutter",
        std::bind(&RunnerCutterControlNode::onStartRunnerCutter, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    startCircleFollowerService_ = create_service<std_srvs::srv::Trigger>(
        "~/start_circle_follower",
        std::bind(&RunnerCutterControlNode::onStartCircleFollower, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    stopService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop",
        std::bind(&RunnerCutterControlNode::onStop, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    getStateService_ =
        create_service<runner_cutter_control_interfaces::srv::GetState>(
            "~/get_state",
            std::bind(&RunnerCutterControlNode::onGetState, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);

    //////////
    // Clients
    //////////
    laser_ = std::make_shared<LaserControlClient>(
        *this, getParamLaserControlNodeName());
    camera_ = std::make_shared<CameraControlClient>(
        *this, getParamCameraControlNodeName());
    detection_ =
        std::make_shared<DetectionClient>(*this, getParamDetectionNodeName());

    calibration_ = std::make_shared<Calibration>(laser_, camera_, detection_);

    // Publish initial state
    publishState();
  }

  ~RunnerCutterControlNode() { stopTask(); }

 private:
#pragma region Param helpers

  std::string getParamLaserControlNodeName() {
    return get_parameter("laser_control_node_name").as_string();
  }

  std::string getParamCameraControlNodeName() {
    return get_parameter("camera_control_node_name").as_string();
  }

  std::string getParamDetectionNodeName() {
    return get_parameter("detection_node_name").as_string();
  }

  std::pair<int, int> getParamCalibrationGridSize() {
    auto param{get_parameter("calibration_grid_size").as_integer_array()};
    return {static_cast<int>(param[0]), static_cast<int>(param[1])};
  }

  std::pair<float, float> getParamCalibrationXBounds() {
    auto param{get_parameter("calibration_x_bounds").as_double_array()};
    return {static_cast<float>(param[0]), static_cast<float>(param[1])};
  }

  std::pair<float, float> getParamCalibrationYBounds() {
    auto param{get_parameter("calibration_y_bounds").as_double_array()};
    return {static_cast<float>(param[0]), static_cast<float>(param[1])};
  }

  LaserColor getParamTrackingLaserColor() {
    auto param{get_parameter("tracking_laser_color").as_double_array()};
    return {static_cast<float>(param[0]), static_cast<float>(param[1]),
            static_cast<float>(param[2]), static_cast<float>(param[3])};
  }

  LaserColor getParamBurnLaserColor() {
    auto param{get_parameter("burn_laser_color").as_double_array()};
    return {static_cast<float>(param[0]), static_cast<float>(param[1]),
            static_cast<float>(param[2]), static_cast<float>(param[3])};
  }

  float getParamBurnTimeSecs() {
    return static_cast<float>(get_parameter("burn_time_secs").as_double());
  }

  int getParamTargetAttempts() {
    return static_cast<int>(get_parameter("target_attempts").as_int());
  }

  float getParamAutoDisarmSecs() {
    return static_cast<float>(get_parameter("auto_disarm_secs").as_double());
  }

  float getParamTrackMissTimeoutSecs() {
    return static_cast<float>(
        get_parameter("track_miss_timeout_secs").as_double());
  }

  std::string getParamSaveDir() {
    return get_parameter("save_dir").as_string();
  }

  bool getParamEnableRunnerCutterV2() {
    return get_parameter("enable_runner_cutter_v2").as_bool();
  }

  float getParamLookaheadSecs() {
    return static_cast<float>(get_parameter("lookahead_secs").as_double());
  }

#pragma endregion

#pragma region State and notifs publishing

  runner_cutter_control_interfaces::msg::State::UniquePtr getStateMsg() {
    std::lock_guard<std::mutex> lock(taskMutex_);

    auto msg{std::make_unique<runner_cutter_control_interfaces::msg::State>()};
    msg->calibrated = calibration_->isCalibrated();
    msg->state = taskRunning_ ? taskName_ : "idle";
    auto [minX, minY, width, height]{calibration_->getNormalizedLaserBounds()};
    common_interfaces::msg::Vector4 normalizedLaserBoundsMsg;
    normalizedLaserBoundsMsg.w = minX;
    normalizedLaserBoundsMsg.x = minY;
    normalizedLaserBoundsMsg.y = width;
    normalizedLaserBoundsMsg.z = height;
    msg->normalized_laser_bounds = normalizedLaserBoundsMsg;
    return msg;
  }

  void publishState() { statePublisher_->publish(std::move(getStateMsg())); }

  void publishNotification(
      const std::string& msg,
      rclcpp::Logger::Level level = rclcpp::Logger::Level::Info) {
    common::publishNotification(get_logger(), notificationsPublisher_, msg,
                                level);
  }

#pragma endregion

#pragma region Callbacks

  void onLaserState(const laser_control_interfaces::msg::State::SharedPtr msg) {
    // Failsafe - stop current task if laser is disconnected
    if (msg->device_state ==
            laser_control_interfaces::msg::DeviceState::DISCONNECTED ||
        msg->device_state ==
            laser_control_interfaces::msg::DeviceState::CONNECTING) {
      bool res{stopTask()};
      if (res) {
        publishNotification("Laser disconnected. Task stopped");
      }
    }
  }

  void onCameraState(
      const camera_control_interfaces::msg::State::SharedPtr msg) {
    // Failsafe - stop current task if camera is disconnected
    if (msg->device_state ==
            camera_control_interfaces::msg::DeviceState::DISCONNECTED ||
        msg->device_state ==
            camera_control_interfaces::msg::DeviceState::CONNECTING) {
      bool res{stopTask()};
      if (res) {
        publishNotification("Camera disconnected. Task stopped");
      }
    }
  }

  void onCalibrate(
      const std::shared_ptr<
          runner_cutter_control_interfaces::srv::Calibrate::Request>
          request,
      std::shared_ptr<
          runner_cutter_control_interfaces::srv::Calibrate::Response>
          response) {
    bool saveImages{request->save_images};
    bool res{startTask("calibration", [this, saveImages]() {
      CalibrationTask task{calibration_, get_logger(), notificationsPublisher_};
      task.run(saveImages, getParamTrackingLaserColor(),
               getParamCalibrationGridSize(), getParamCalibrationXBounds(),
               getParamCalibrationYBounds(), taskStopSignal_);
    })};
    response->success = res;
  }

  void onSaveCalibration(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::string filePath{common::expandUser(getParamSaveDir()) +
                         "/calibration.dat"};
    bool res{calibration_->save(filePath)};
    if (res) {
      publishNotification(fmt::format("Calibration saved: {}", filePath));
    } else {
      publishNotification("Calibration could not be saved",
                          rclcpp::Logger::Level::Warn);
    }
    response->success = res;
  }

  void onLoadCalibration(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::string filePath{common::expandUser(getParamSaveDir()) +
                         "/calibration.dat"};
    bool res{calibration_->load(filePath)};
    if (res) {
      publishNotification(fmt::format("Calibration loaded: {}", filePath));
      publishState();
    } else {
      publishNotification(
          "Calibration file does not exist or could not be loaded",
          rclcpp::Logger::Level::Warn);
    }
    response->success = res;
  }

  void onAddCalibrationPoints(
      const std::shared_ptr<
          runner_cutter_control_interfaces::srv::AddCalibrationPoints::Request>
          request,
      std::shared_ptr<
          runner_cutter_control_interfaces::srv::AddCalibrationPoints::Response>
          response) {
    std::vector<NormalizedPixelCoord> normalizedPixelCoords;
    for (const auto& coord : request->normalized_pixel_coords) {
      normalizedPixelCoords.push_back(NormalizedPixelCoord{
          static_cast<float>(coord.x), static_cast<float>(coord.y)});
    }
    bool saveImages{request->save_images};
    bool res{startTask(
        "add_calibration_points",
        [this, normalizedPixelCoords = std::move(normalizedPixelCoords),
         saveImages]() {
          AddCalibrationPointsTask task{detection_, calibration_, get_logger(),
                                        notificationsPublisher_};
          task.run(normalizedPixelCoords, saveImages,
                   getParamTrackingLaserColor(), taskStopSignal_);
        })};
    response->success = res;
  }

  void onManualTargetLaser(
      const std::shared_ptr<
          runner_cutter_control_interfaces::srv::ManualTargetLaser::Request>
          request,
      std::shared_ptr<
          runner_cutter_control_interfaces::srv::ManualTargetLaser::Response>
          response) {
    NormalizedPixelCoord normalizedPixelCoord{
        static_cast<float>(request->normalized_pixel_coord.x),
        static_cast<float>(request->normalized_pixel_coord.y)};
    bool shouldAim{request->aim};
    bool shouldBurn{request->burn};
    bool res{startTask("manual_target_laser", [this, normalizedPixelCoord,
                                               shouldAim, shouldBurn]() {
      ManualTargetLaserTask task{laser_, camera_, detection_, calibration_,
                                 get_logger()};
      task.run(normalizedPixelCoord, shouldAim, shouldBurn,
               getParamTrackingLaserColor(), getParamBurnLaserColor(),
               getParamBurnTimeSecs(), taskStopSignal_);
    })};
    response->success = res;
  }

  void onStartRunnerCutter(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    bool res{startTask("runner_cutter", [this]() {
      if (getParamEnableRunnerCutterV2()) {
        RunnerCutterTask task{detectionCallbackRegistry_, laser_, detection_,
                              calibration_, get_logger()};
        task.run(getParamTrackMissTimeoutSecs(), getParamTargetAttempts(),
                 getParamLookaheadSecs(), getParamBurnLaserColor(),
                 getParamBurnTimeSecs(), taskStopSignal_);
      } else {
        StaticRunnerCutterTask task{detectionCallbackRegistry_,
                                    camera_,
                                    detection_,
                                    calibration_,
                                    laser_,
                                    get_logger(),
                                    notificationsPublisher_,
                                    tracksPublisher_};
        task.run(getParamTrackMissTimeoutSecs(), getParamTargetAttempts(),
                 /*enableDetectionDuringBurn=*/false, /*enableAiming=*/true,
                 getParamAutoDisarmSecs(), getParamSaveDir(),
                 getParamTrackingLaserColor(), getParamBurnLaserColor(),
                 getParamBurnTimeSecs(), taskStopSignal_);
      }
    })};
    response->success = res;
  }

  void onStartCircleFollower(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    bool res{startTask("circle_follower", [this]() {
      CircleFollowerTask task{detectionCallbackRegistry_, laser_, detection_,
                              calibration_, get_logger()};
      task.run(getParamTrackMissTimeoutSecs(), getParamTargetAttempts(),
               getParamLookaheadSecs(), getParamTrackingLaserColor(),
               /*laserIntervalSecs=*/0.25f, taskStopSignal_);
    })};
    response->success = res;
  }

  void onStop(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
              std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    bool res{stopTask()};
    response->success = res;
  }

  void onGetState(
      const std::shared_ptr<
          runner_cutter_control_interfaces::srv::GetState::Request>,
      std::shared_ptr<runner_cutter_control_interfaces::srv::GetState::Response>
          response) {
    response->state = std::move(*getStateMsg());
  }

#pragma endregion

#pragma region Task management

  void resetToIdle() {
    laser_->clearPaths();
    laser_->stop();
    detection_->stopAllDetections();
    detectionCallbackRegistry_->clear();
  }

  bool startTask(const std::string& taskName, std::function<void()> taskFunc) {
    std::unique_lock<std::mutex> lock(taskMutex_);

    // If a task is already running, don't start another task
    if (taskRunning_) {
      return false;
    }

    // If the task is done, but the thread has not been joined yet, do it now
    if (taskThread_.joinable()) {
      lock.unlock();  // unlock before joining to prevent deadlock
      taskThread_.join();
      lock.lock();
    }

    taskStopSignal_ = false;
    taskName_ = taskName;
    taskRunning_ = true;

    taskThread_ = std::thread([this, taskFunc = std::move(taskFunc)]() {
      try {
        resetToIdle();
        publishState();
        taskFunc();
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Task exception: %s", e.what());
      }

      resetToIdle();

      {
        std::lock_guard<std::mutex> lock(taskMutex_);
        taskRunning_ = false;
      }

      publishState();
    });

    return true;
  }

  bool stopTask() {
    std::unique_lock<std::mutex> lock(taskMutex_);

    if (!taskThread_.joinable()) {
      return false;
    }

    taskStopSignal_ = true;
    lock.unlock();  // unlock before joining to prevent deadlock
    taskThread_.join();
    lock.lock();
    return true;
  }

#pragma endregion

  rclcpp::Publisher<runner_cutter_control_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::Publisher<runner_cutter_control_interfaces::msg::Tracks>::SharedPtr
      tracksPublisher_;
  rclcpp::CallbackGroup::SharedPtr subscriberCallbackGroup_;
  rclcpp::Subscription<laser_control_interfaces::msg::State>::SharedPtr
      laserStateSubscriber_;
  rclcpp::Subscription<camera_control_interfaces::msg::State>::SharedPtr
      cameraStateSubscriber_;
  rclcpp::CallbackGroup::SharedPtr detectionsCallbackGroup_;
  rclcpp::Subscription<detection_interfaces::msg::DetectionResult>::SharedPtr
      detectionsSubscriber_;
  std::shared_ptr<CallbackRegistry<detection_interfaces::msg::DetectionResult>>
      detectionCallbackRegistry_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<runner_cutter_control_interfaces::srv::Calibrate>::SharedPtr
      calibrateService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr saveCalibrationService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr loadCalibrationService_;
  rclcpp::Service<runner_cutter_control_interfaces::srv::AddCalibrationPoints>::
      SharedPtr addCalibrationPointsService_;
  rclcpp::Service<runner_cutter_control_interfaces::srv::ManualTargetLaser>::
      SharedPtr manualTargetLaserService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr startRunnerCutterService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
      startCircleFollowerService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopService_;
  rclcpp::Service<runner_cutter_control_interfaces::srv::GetState>::SharedPtr
      getStateService_;

  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<CameraControlClient> camera_;
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  std::thread taskThread_;
  std::mutex taskMutex_;
  std::atomic<bool> taskStopSignal_{false};
  std::atomic<bool> taskRunning_{false};
  std::string taskName_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  // MultiThreadedExecutor allows callbacks to run in parallel
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node{std::make_shared<RunnerCutterControlNode>()};
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();

  return 0;
}

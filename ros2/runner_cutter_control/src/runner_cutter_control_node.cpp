#include <fmt/core.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>

#include "camera_control_interfaces/msg/detection_result.hpp"
#include "camera_control_interfaces/msg/detection_type.hpp"
#include "camera_control_interfaces/msg/device_state.hpp"
#include "camera_control_interfaces/msg/state.hpp"
#include "laser_control_interfaces/msg/device_state.hpp"
#include "laser_control_interfaces/msg/state.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/clients/laser_detection_context.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"
#include "runner_cutter_control_interfaces/msg/state.hpp"
#include "runner_cutter_control_interfaces/msg/track.hpp"
#include "runner_cutter_control_interfaces/msg/track_state.hpp"
#include "runner_cutter_control_interfaces/msg/tracks.hpp"
#include "runner_cutter_control_interfaces/srv/add_calibration_points.hpp"
#include "runner_cutter_control_interfaces/srv/get_state.hpp"
#include "runner_cutter_control_interfaces/srv/manual_target_laser.hpp"
#include "std_srvs/srv/trigger.hpp"

/**
 * Concurrency primitive that provides a shared flag that can be set and waited
 * on.
 */
class Event {
 public:
  Event() : flag_(false) {}

  void set() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = true;
    cv_.notify_all();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = false;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return flag_; });
  }

  bool wait_for(float timeoutSecs) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cv_.wait_for(lock, std::chrono::duration<float>(timeoutSecs),
                        [this] { return flag_; });
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  bool flag_;
};

std::pair<int, int> millisecondsToRosTime(double milliseconds) {
  // ROS timestamps consist of two integers, one for seconds and one for
  // nanoseconds
  int seconds = static_cast<int>(milliseconds / 1000);
  int nanoseconds =
      static_cast<int>((static_cast<int>(milliseconds) % 1000) * 1e6);
  return {seconds, nanoseconds};
}

std::string expandUser(const std::string& path) {
  if (path.empty() || path[0] != '~') {
    return path;
  }

  const char* home = std::getenv("HOME");
  if (!home) {
    throw std::runtime_error("HOME environment variable not set");
  }

  return std::string(home) + path.substr(1);
}

// TODO: stop task if laser or camera are disconnected

class RunnerCutterControlNode : public rclcpp::Node {
 public:
  explicit RunnerCutterControlNode() : Node("runner_cutter_control_node") {
    /////////////
    // Parameters
    /////////////
    declare_parameter<std::string>("laser_control_node_name", "laser0");
    declare_parameter<std::string>("camera_control_node_name", "camera0");
    declare_parameter<std::vector<float>>("tracking_laser_color",
                                          {0.15f, 0.0f, 0.0f, 0.0f});
    declare_parameter<std::vector<float>>("burn_laser_color",
                                          {0.0f, 0.0f, 1.0f, 0.0f});
    declare_parameter<float>("burn_time_secs", 5.0);
    declare_parameter<bool>("enable_aiming", true);
    // Max number of times to attempt to target a detected runner to burn. An
    // attempt may fail if the runner burn point is outside the laser bounds, if
    // the aiming process failed, or if the runner was no longer detected. A
    // negative number means no limit.
    declare_parameter<int>("target_attempts", -1);
    // Duration, in seconds, during which if no viable target becomes available,
    // the runner cutter task will stop. A negative number means no auto disarm.
    declare_parameter<float>("auto_disarm_secs", -1.0);
    declare_parameter<std::string>("save_dir", "~/runner_cutter");

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
    auto cameraDetectionsTopicName{
        fmt::format("/{}/detections", getParamCameraControlNodeName())};
    detectionsSubscriber_ =
        create_subscription<camera_control_interfaces::msg::DetectionResult>(
            cameraDetectionsTopicName, rclcpp::SensorDataQoS(),
            std::bind(&RunnerCutterControlNode::onDetection, this,
                      std::placeholders::_1),
            options);

    ///////////
    // Services
    ///////////
    serviceCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    // TODO: use action instead once there's a new release of roslib. Currently
    // roslib does not support actions with ROS2
    calibrateService_ = create_service<std_srvs::srv::Trigger>(
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

    calibration_ = std::make_shared<Calibration>(laser_, camera_);
    tracker_ = std::make_shared<Tracker>();

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

  std::tuple<float, float, float, float> getParamTrackingLaserColor() {
    auto param{get_parameter("tracking_laser_color").as_double_array()};
    return {param[0], param[1], param[2], param[3]};
  }

  std::tuple<float, float, float, float> getParamBurnLaserColor() {
    auto param{get_parameter("burn_laser_color").as_double_array()};
    return {param[0], param[1], param[2], param[3]};
  }

  float getParamBurnTimeSecs() {
    return static_cast<float>(get_parameter("burn_time_secs").as_double());
  }

  bool getParamEnableAiming() {
    return get_parameter("enable_aiming").as_bool();
  }

  int getParamTargetAttempts() {
    return static_cast<int>(get_parameter("target_attempts").as_int());
  }

  float getParamAutoDisarmSecs() {
    return static_cast<float>(get_parameter("auto_disarm_secs").as_double());
  }

  std::string getParamSaveDir() {
    return get_parameter("save_dir").as_string();
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

  runner_cutter_control_interfaces::msg::Tracks::UniquePtr getTracksMsg() {
    auto msg{std::make_unique<runner_cutter_control_interfaces::msg::Tracks>()};
    auto [frameWidth, frameHeight]{calibration_->getCameraFrameSize()};
    for (const auto& [id, track] : tracker_->getTracks()) {
      runner_cutter_control_interfaces::msg::Track trackMsg;
      trackMsg.id = track->getId();
      common_interfaces::msg::Vector2 normalizedPixelCoordMsg;
      normalizedPixelCoordMsg.x =
          frameWidth > 0 ? static_cast<float>(track->getPixel().first) /
                               static_cast<float>(frameWidth)
                         : -1.0f;
      normalizedPixelCoordMsg.y =
          frameHeight > 0 ? static_cast<float>(track->getPixel().second) /
                                static_cast<float>(frameHeight)
                          : -1.0f;
      trackMsg.normalized_pixel_coord = normalizedPixelCoordMsg;
      switch (track->getState()) {
        case Track::State::PENDING:
          trackMsg.state =
              runner_cutter_control_interfaces::msg::TrackState::PENDING;
          break;
        case Track::State::ACTIVE:
          trackMsg.state =
              runner_cutter_control_interfaces::msg::TrackState::ACTIVE;
          break;
        case Track::State::COMPLETED:
          trackMsg.state =
              runner_cutter_control_interfaces::msg::TrackState::COMPLETED;
          break;
        case Track::State::FAILED:
          trackMsg.state =
              runner_cutter_control_interfaces::msg::TrackState::FAILED;
          break;
      }
      msg->tracks.push_back(trackMsg);
    }
    return msg;
  }

  void publishState() { statePublisher_->publish(std::move(getStateMsg())); }

  void publishNotification(
      const std::string& msg,
      rclcpp::Logger::Level level = rclcpp::Logger::Level::Info) {
    uint8_t logMsgLevel = 0;
    switch (level) {
      case rclcpp::Logger::Level::Debug:
        RCLCPP_DEBUG(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::DEBUG;
        break;
      case rclcpp::Logger::Level::Info:
        RCLCPP_INFO(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::INFO;
        break;
      case rclcpp::Logger::Level::Warn:
        RCLCPP_WARN(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::WARN;
        break;
      case rclcpp::Logger::Level::Error:
        RCLCPP_ERROR(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::ERROR;
        break;
      case rclcpp::Logger::Level::Fatal:
        RCLCPP_FATAL(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::FATAL;
        break;
      default:
        RCLCPP_ERROR(get_logger(), "Unknown log level: %s", msg.c_str());
        return;
    }

    // Get current time in milliseconds
    double timestampMillis{
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count()) /
        1000.0};
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto logMsg{rcl_interfaces::msg::Log()};
    logMsg.stamp.sec = sec;
    logMsg.stamp.nanosec = nanosec;
    logMsg.level = logMsgLevel;
    logMsg.msg = msg;
    notificationsPublisher_->publish(std::move(logMsg));
  }

  void publishTracks() { tracksPublisher_->publish(std::move(getTracksMsg())); }

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

  void onDetection(
      const camera_control_interfaces::msg::DetectionResult::SharedPtr msg) {
    if (msg->detection_type ==
            camera_control_interfaces::msg::DetectionType::RUNNER ||
        msg->detection_type ==
            camera_control_interfaces::msg::DetectionType::CIRCLE) {
      // For new tracks, add to tracker and set as pending. For tracks that are
      // detected again, update the track pixel and position; for FAILED tracks,
      // set them as PENDING since they may have moved since the last detection.
      std::unordered_set<int> prevPendingTracks;
      for (const auto& track :
           tracker_->getTracksWithState(Track::State::PENDING)) {
        prevPendingTracks.insert(track->getId());
      }

      std::unordered_set<int> prevDetectedTrackIds{lastDetectedTrackIds_};
      lastDetectedTrackIds_.clear();

      for (const auto& instance : msg->instances) {
        std::pair<int, int> pixel{
            static_cast<int>(std::round(instance.point.x)),
            static_cast<int>(std::round(instance.point.y))};
        std::tuple<float, float, float> position{
            instance.position.x, instance.position.y, instance.position.z};

        std::shared_ptr<Track> track;
        try {
          track =
              tracker_->addTrack(instance.track_id, pixel, position,
                                 msg->timestamp * 1000.0, instance.confidence);
        } catch (const std::exception& e) {
          continue;
        }

        lastDetectedTrackIds_.insert(instance.track_id);

        // Put detected tracks that are marked as failed back into the pending
        // queue, since we want to reattempt to burn them (up to targetAttempts
        // times) as they could now potentially be in bounds.
        int numAttempts{getParamTargetAttempts()};
        if (track->getState() == Track::State::FAILED &&
            (numAttempts < 0 ||
             track->getStateCount(Track::State::FAILED) < numAttempts)) {
          tracker_->processTrack(track->getId(), Track::State::PENDING);
        }
      }

      // Mark as FAILED any tracks that were previously detected, are PENDING,
      // but are no longer detected.
      for (int trackId : prevDetectedTrackIds) {
        if (lastDetectedTrackIds_.find(trackId) ==
            lastDetectedTrackIds_.end()) {
          // We found a track that was previously detected but not anymore
          auto trackOpt{tracker_->getTrack(trackId)};
          if (!trackOpt) {
            continue;
          }

          auto track{trackOpt.value()};
          track->setPixel({-1, -1});
          track->setPosition({-1.0f, -1.0f, -1.0f});
          if (track->getState() == Track::State::PENDING) {
            tracker_->processTrack(trackId, Track::State::FAILED);
          }
        }
      }

      // Notify when the pending tracks have changed
      std::unordered_set<int> pendingTracks;
      for (const auto& track :
           tracker_->getTracksWithState(Track::State::PENDING)) {
        pendingTracks.insert(track->getId());
      }
      if (prevPendingTracks != pendingTracks) {
        pendingTracksChangedEvent_.set();
      }
    }
  }

  void onCalibrate(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    bool res{
        startTask("calibration", &RunnerCutterControlNode::calibrationTask)};
    response->success = res;
  }

  void onSaveCalibration(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::string filePath{expandUser(getParamSaveDir()) + "/calibration.dat"};
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
    std::string filePath{expandUser(getParamSaveDir()) + "/calibration.dat"};
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
    auto normalizedPixelCoords{
        std::make_shared<std::vector<std::pair<float, float>>>()};
    for (const auto& coord : request->normalized_pixel_coords) {
      normalizedPixelCoords->push_back({coord.x, coord.y});
    }
    bool res{startTask("add_calibration_points",
                       &RunnerCutterControlNode::addCalibrationPointsTask,
                       normalizedPixelCoords)};
    response->success = res;
  }

  void onManualTargetLaser(
      const std::shared_ptr<
          runner_cutter_control_interfaces::srv::ManualTargetLaser::Request>
          request,
      std::shared_ptr<
          runner_cutter_control_interfaces::srv::ManualTargetLaser::Response>
          response) {
    std::pair<float, float> normalizedPixelCoord{
        request->normalized_pixel_coord.x, request->normalized_pixel_coord.y};
    bool res{startTask("manual_target_laser",
                       &RunnerCutterControlNode::manualTargetLaserTask,
                       normalizedPixelCoord, request->aim, request->burn)};
    response->success = res;
  }

  void onStartRunnerCutter(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    bool res{startTask(
        "runner_cutter", &RunnerCutterControlNode::runnerCutterTask,
        camera_control_interfaces::msg::DetectionType::RUNNER, false)};
    response->success = res;
  }

  void onStartCircleFollower(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    bool res{startTask("circle_follower",
                       &RunnerCutterControlNode::circleFollowerTask, 0.5f)};
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
    laser_->clearPoint();
    laser_->stop();
    camera_->stopAllDetections();
    tracker_->clear();
    lastDetectedTrackIds_.clear();
  }

  template <typename Function, typename... Args>
  bool startTask(const std::string& taskName, Function&& func, Args&&... args) {
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

    // Store arguments in a tuple for C++17 compatibility
    auto taskArgs{std::make_tuple(std::forward<Args>(args)...)};

    taskThread_ =
        std::thread([this, func = std::forward<Function>(func), taskArgs]() {
          try {
            resetToIdle();
            publishState();
            std::apply(
                [this, &func](auto&&... unpackedArgs) {
                  std::invoke(
                      func, this,
                      std::forward<decltype(unpackedArgs)>(unpackedArgs)...);
                },
                taskArgs);
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
    pendingTracksChangedEvent_.set();
    lock.unlock();  // unlock before joining to prevent deadlock
    taskThread_.join();
    lock.lock();
    return true;
  }

#pragma endregion

#pragma region Task definitions

  void calibrationTask() {
    publishNotification("Calibration started");
    calibration_->calibrate(getParamTrackingLaserColor(), {5, 5},
                            taskStopSignal_);
    publishNotification("Calibration complete");
  }

  void addCalibrationPointsTask(
      std::shared_ptr<std::vector<std::pair<float, float>>>
          normalizedPixelCoords) {
    // For each camera pixel coord, find the 3D position wrt the camera
    auto positionsOpt{camera_->getPositions(*normalizedPixelCoords)};
    if (!positionsOpt) {
      return;
    }
    auto positions{positionsOpt.value()};

    // Filter out any invalid positions (x, y, and z are all negative)
    positions.erase(
        std::remove_if(positions.begin(), positions.end(),
                       [](const std::tuple<float, float, float>& position) {
                         auto [x, y, z]{position};
                         return x < 0.0f && y < 0.0f && z < 0.0f;
                       }),
        positions.end());

    // Convert camera positions to laser pixels
    std::vector<std::pair<float, float>> laserCoords;
    for (const auto& position : positions) {
      laserCoords.push_back(calibration_->cameraPositionToLaserCoord(position));
    }

    // Filter out laser coords that are out of bounds
    laserCoords.erase(std::remove_if(laserCoords.begin(), laserCoords.end(),
                                     [](const std::pair<float, float>& coord) {
                                       auto [x, y]{coord};
                                       return !(0.0f <= x && x <= 1.0f &&
                                                0.0f <= y && y <= 1.0f);
                                     }),
                      laserCoords.end());

    std::size_t numPointsAdded{calibration_->addCalibrationPoints(
        laserCoords, getParamTrackingLaserColor(), true, taskStopSignal_)};

    publishNotification(
        fmt::format("Added {} calibration point(s)", numPointsAdded));
  }

  void manualTargetLaserTask(std::pair<float, float> normalizedPixelCoord,
                             bool shouldAim, bool shouldBurn) {
    // Find the 3D position wrt the camera
    std::vector<std::pair<float, float>> normalizedPixelCoords{
        normalizedPixelCoord};
    auto positionsOpt{camera_->getPositions(normalizedPixelCoords)};
    if (!positionsOpt) {
      return;
    }

    auto positions{positionsOpt.value()};
    auto targetPosition{positions[0]};
    auto [frameWidth, frameHeight]{calibration_->getCameraFrameSize()};
    std::pair<int, int> targetPixel{
        static_cast<int>(std::round(normalizedPixelCoord.first * frameWidth)),
        static_cast<int>(
            std::round(normalizedPixelCoord.second * frameHeight))};

    // Aim
    std::pair<float, float> laserCoord;
    if (shouldAim) {
      auto laserCoordOpt{aim(targetPosition, targetPixel)};
      if (!laserCoordOpt) {
        RCLCPP_INFO(get_logger(), "Failed to aim laser");
        return;
      }
      RCLCPP_INFO(get_logger(), "Aim laser successful");
      laserCoord = laserCoordOpt.value();
    } else {
      laserCoord = calibration_->cameraPositionToLaserCoord(targetPosition);
    }

    // Burn
    if (shouldBurn) {
      burnTarget(-1, laserCoord);
    }
  }

  void runnerCutterTask(
      uint8_t detectionType =
          camera_control_interfaces::msg::DetectionType::RUNNER,
      bool enableDetectionDuringBurn = false) {
    publishTracks();

    auto timestamp{
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
    std::stringstream datetimeString;
    datetimeString << std::put_time(std::localtime(&timestamp), "%Y%m%d%H%M%S");
    std::string runDataDir{
        fmt::format("{}/runs/{}", getParamSaveDir(), datetimeString.str())};
    camera_->setSaveDirectory(runDataDir);
    camera_->saveImage();

    // Start runner detection with detection bounds set to the laser's FOV.
    // Note: The ML model will still detect runners and assign instance IDs
    // using the full color camera frame, but if the runner is completely out of
    // the detection bounds, the result is not published via the detections
    // topic.
    auto normalizedLaserBounds{calibration_->getNormalizedLaserBounds()};
    camera_->startDetection(detectionType, normalizedLaserBounds);

    while (!taskStopSignal_) {
      // Attempt to acquire target
      auto targetOpt{acquireNextTarget()};
      publishTracks();

      // If there are no valid targets, wait for another detection event
      if (!targetOpt) {
        RCLCPP_INFO(get_logger(), "No targets found. Waiting for detection.");
        float timeoutSecs{getParamAutoDisarmSecs()};
        if (timeoutSecs > 0.0f) {
          // End task if no new valid targets for timeoutSecs
          if (!pendingTracksChangedEvent_.wait_for(timeoutSecs)) {
            RCLCPP_INFO(
                get_logger(),
                "No new targets after %f second(s). Ending runner cutter task.",
                timeoutSecs);
            break;
          }

        } else {
          pendingTracksChangedEvent_.wait();
        }
        pendingTracksChangedEvent_.clear();
        continue;
      }

      auto target{targetOpt.value()};
      if (!enableDetectionDuringBurn) {
        // Temporarily disable runner detection during aim/burn
        camera_->stopDetection(detectionType);
      }

      // Aim
      std::pair<float, float> laserCoord;
      if (getParamEnableAiming()) {
        auto laserCoordOpt{aim(target->getPosition(), target->getPixel())};
        if (!laserCoordOpt) {
          RCLCPP_INFO(get_logger(), "Failed to aim laser at track %d.",
                      target->getId());
          tracker_->processTrack(target->getId(), Track::State::FAILED);
          continue;
        }
        laserCoord = laserCoordOpt.value();
      } else {
        laserCoord =
            calibration_->cameraPositionToLaserCoord(target->getPosition());
      }

      // Burn
      burnTarget(target->getId(), laserCoord);

      if (!enableDetectionDuringBurn) {
        camera_->startDetection(detectionType);
      }
    }
  }

  /**
   * Get the next suitable target from the tracker.
   *
   * @return The target Track, if one is available.
   */
  std::optional<std::shared_ptr<Track>> acquireNextTarget() {
    auto activeTracks{tracker_->getTracksWithState(Track::State::ACTIVE)};
    if (!activeTracks.empty()) {
      RCLCPP_INFO(get_logger(), "Using active track [%d]",
                  activeTracks[0]->getId());
      return activeTracks[0];
    }

    while (!taskStopSignal_) {
      auto trackOpt{tracker_->getNextPendingTrack()};
      if (!trackOpt) {
        return std::nullopt;
      }

      auto track{trackOpt.value()};
      RCLCPP_INFO(get_logger(), "Processing pending track [%d]",
                  track->getId());

      auto laserCoord{
          calibration_->cameraPositionToLaserCoord(track->getPosition())};
      if (laserCoord.first >= 0.0 && laserCoord.first <= 1.0 &&
          laserCoord.second >= 0.0 && laserCoord.second <= 1.0) {
        RCLCPP_INFO(get_logger(), "Setting track [%d] as target.",
                    track->getId());
        return track;
      }

      RCLCPP_INFO(get_logger(), "Track [%d] out of bounds. Marking as failed.",
                  track->getId());
      tracker_->processTrack(track->getId(), Track::State::FAILED);
    }

    return std::nullopt;
  }

  void burnTarget(int targetTrackId, std::pair<float, float> laserCoord) {
    float burnTimeSecs{getParamBurnTimeSecs()};
    laser_->clearPoint();
    auto [r, g, b, i]{getParamBurnLaserColor()};
    laser_->setColor(r, g, b, i);
    laser_->play();
    RCLCPP_INFO(get_logger(), "Burning track %d for %f secs...", targetTrackId,
                burnTimeSecs);
    laser_->setPoint(laserCoord.first, laserCoord.second);
    std::this_thread::sleep_for(std::chrono::duration<float>(burnTimeSecs));
    laser_->clearPoint();
    laser_->stop();
    tracker_->processTrack(targetTrackId, Track::State::COMPLETED);
    RCLCPP_INFO(get_logger(), "Burn complete on track %d...", targetTrackId);
  }

  void circleFollowerTask(float laserIntervalSecs = 0.5f) {
    laser_->clearPoint();
    auto [r, g, b, i]{getParamTrackingLaserColor()};
    laser_->setColor(r, g, b, i);
    laser_->play();
    camera_->startDetection(
        camera_control_interfaces::msg::DetectionType::CIRCLE);

    while (!taskStopSignal_) {
      std::this_thread::sleep_for(
          std::chrono::duration<float>(laserIntervalSecs));
      // Follow mode currently only supports a single target with ID 1
      auto trackOpt{tracker_->getTrack(1)};
      if (!trackOpt) {
        continue;
      }

      auto track{trackOpt.value()};

      // Fire tracking laser at target using predicted future position
      // TODO: Determine actual latency
      double estimatedCameraLatencyMs{100.0};
      double timestampMillis{
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count()};
      auto predictedPosition{track->getPredictor().predict(
          timestampMillis + estimatedCameraLatencyMs)};
      auto laserCoord{
          calibration_->cameraPositionToLaserCoord(predictedPosition)};
      track->getPredictor().reset();

      laser_->setPoint(laserCoord.first, laserCoord.second);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      laser_->clearPoint();
    }

    laser_->clearPoint();
    laser_->stop();
    camera_->stopAllDetections();
  }

  /**
   * Attempt to incrementally guide the laser to a target camera pixel. The
   * target camera position is used to calculate the initial laser coordinate,
   * from which incremental corrections are applied until the laser reaches
   * the target camera pixel.
   *
   * @param targetCameraPosition Camera-space position of the target.
   * @param targetCameraPixel Camera pixel coordinate of the target.
   * @return The corrected laser coordinate that projects to the target camera
   * pixel.
   */
  std::optional<std::pair<float, float>> aim(
      std::tuple<float, float, float> targetCameraPosition,
      std::pair<int, int> targetCameraPixel) {
    LaserDetectionContext context{laser_, camera_};
    auto initialLaserCoord{
        calibration_->cameraPositionToLaserCoord(targetCameraPosition)};
    auto [r, g, b, i]{getParamTrackingLaserColor()};
    laser_->setColor(r, g, b, i);
    auto correctedLaserCoord{
        correctLaser(initialLaserCoord, targetCameraPixel)};
    return correctedLaserCoord;
  }

  /**
   * Find a laser coordinate that projects to the target camera pixel
   * coordinate to within a specified pixel distance. Starts from an initial
   * coordinate and incrementally calculates laser coordinates to attempt to
   * get within the threshold distance.
   *
   * @param initialLaserCoord Initial laser coordinate.
   * @param targetCameraPixel Target camera pixel coordinate.
   * @param pixelDistanceThreshold Pixel distance threshold under which the
   * corrected laser coordinate is considered close enough to the target.
   * @return The corrected laser coordinate that projects to the target camera
   * pixel.
   */
  std::optional<std::pair<float, float>> correctLaser(
      std::pair<float, float> initialLaserCoord,
      std::pair<int, int> targetCameraPixel,
      float pixelDistanceThreshold = 2.5f) {
    auto currentLaserCoord{initialLaserCoord};

    while (!taskStopSignal_) {
      laser_->setPoint(currentLaserCoord.first, currentLaserCoord.second);
      // Get detected camera pixel coord and camera-space position for laser
      auto detectResultOpt{detectLaser()};
      if (!detectResultOpt) {
        RCLCPP_WARN(get_logger(), "Could not detect laser during correction");
        return std::nullopt;
      }

      // Calculate camera pixel distance
      auto [laserPixel, laserPosition]{detectResultOpt.value()};
      float dist{static_cast<float>(
          std::hypot(laserPixel.first - targetCameraPixel.first,
                     laserPixel.second - targetCameraPixel.second))};
      RCLCPP_INFO(
          get_logger(),
          "Aiming laser. Target camera pixel = (%d, %d), laser detected at = "
          "(%d, %d), dist = %f",
          targetCameraPixel.first, targetCameraPixel.second,
          targetCameraPixel.first, targetCameraPixel.second, dist);

      if (dist <= pixelDistanceThreshold) {
        RCLCPP_INFO(get_logger(), "Correction successful");
        return currentLaserCoord;
      }

      // Use this opportunity to add to calibration points since we have the
      // laser coord and associated position in camera space
      calibration_->addPointCorrespondence(currentLaserCoord, laserPixel,
                                           laserPosition);

      // Compute correction factor
      float correctionX{static_cast<float>(targetCameraPixel.first) -
                        static_cast<float>(laserPixel.first)};
      float correctionY{static_cast<float>(targetCameraPixel.second) -
                        static_cast<float>(laserPixel.second)};

      // Scale the correction by the camera frame size
      auto [frameWidth, frameHeight]{calibration_->getCameraFrameSize()};
      correctionX /= frameWidth;
      correctionY /= frameHeight;

      // Invert Y coordinate as laser coord Y is flipped from camera frame Y
      correctionY *= -1;

      // Calculate new laser coord
      std::pair<float, float> newLaserCoord{
          currentLaserCoord.first + correctionX,
          currentLaserCoord.second + correctionY};
      RCLCPP_INFO(
          get_logger(),
          "Distance too large. Correcting laser. Current laser coord = (%f, "
          "%f), corrected laser coord = (%f, %f)",
          currentLaserCoord.first, currentLaserCoord.second,
          newLaserCoord.first, newLaserCoord.second);

      if (newLaserCoord.first > 1.0f || newLaserCoord.second > 1.0f ||
          newLaserCoord.first < 0.0f || newLaserCoord.second < 0.0f) {
        RCLCPP_INFO(get_logger(), "Laser coord is outside of renderable area.");
        return std::nullopt;
      }

      currentLaserCoord = newLaserCoord;
    }

    return std::nullopt;
  }

  struct DetectLaserResult {
    std::pair<int, int> cameraPixel;
    std::tuple<float, float, float> cameraPosition;
  };
  std::optional<DetectLaserResult> detectLaser(int maxAttempts = 3) {
    int attempt{0};
    while (attempt < maxAttempts && !taskStopSignal_) {
      auto detectionResult{camera_->getDetection(
          camera_control_interfaces::msg::DetectionType::LASER, true)};
      auto instances{detectionResult->instances};
      if (instances.size() > 0) {
        // TODO: better handle case where more than 1 laser detected. For now,
        // just ignore everything except the first result.
        if (instances.size() > 1) {
          RCLCPP_WARN(get_logger(),
                      "Found more than 1 laser during correction");
        }
        auto instance{instances[0]};
        return DetectLaserResult{
            {static_cast<int>(std::round(instance.point.x)),
             static_cast<int>(std::round(instance.point.y))},
            {instance.position.x, instance.position.y, instance.position.z}};
      }

      // No lasers detected. Try again.
      ++attempt;
    }

    return std::nullopt;
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
  rclcpp::Subscription<camera_control_interfaces::msg::DetectionResult>::
      SharedPtr detectionsSubscriber_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr calibrateService_;
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
  std::shared_ptr<Calibration> calibration_;
  std::thread taskThread_;
  std::mutex taskMutex_;
  std::atomic<bool> taskStopSignal_{false};
  std::atomic<bool> taskRunning_{false};
  std::string taskName_;
  std::shared_ptr<Tracker> tracker_;
  std::unordered_set<int> lastDetectedTrackIds_;
  // Notifies waiting threads that new pending tracks were detected
  Event pendingTracksChangedEvent_;
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
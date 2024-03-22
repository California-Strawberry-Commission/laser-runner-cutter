import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";

export default function useCameraNode(nodeName: string) {
  const ros = useContext(ROSContext);
  const [nodeState, setNodeState] = useState({
    connected: false,
    laser_detection_enabled: false,
    runner_detection_enabled: false,
    recording_video: false,
  });
  const [frameSrc, setFrameSrc] = useState<string>("");

  // Initial node state
  useEffect(() => {
    const getState = async () => {
      const result = await ros.callService(
        `${nodeName}/get_state`,
        "camera_control_interfaces/GetState",
        {}
      );
      setNodeState(result.state);
    };
    getState();
  }, [nodeName]);

  // Subscriptions
  useEffect(() => {
    const stateSub = ros.subscribe(
      `${nodeName}/state`,
      "camera_control_interfaces/State",
      (message) => {
        setNodeState(message);
      }
    );
    const frameSub = ros.subscribe(
      `${nodeName}/debug_frame`,
      "sensor_msgs/CompressedImage",
      (message) => {
        setFrameSrc(`data:image/jpeg;base64,${message.data}`);
      }
    );

    return () => {
      stateSub.unsubscribe();
      frameSub.unsubscribe();
    };
  }, [nodeName]);

  const setExposure = (exposureMs: number) => {
    ros.callService(
      `${nodeName}/set_exposure`,
      "camera_control_interfaces/SetExposure",
      { exposure_ms: exposureMs }
    );
  };

  const startLaserDetection = () => {
    ros.callService(
      `${nodeName}/start_laser_detection`,
      "std_srvs/Trigger",
      {}
    );
  };

  const stopLaserDetection = () => {
    ros.callService(`${nodeName}/stop_laser_detection`, "std_srvs/Trigger", {});
  };

  const startRunnerDetection = () => {
    ros.callService(
      `${nodeName}/start_runner_detection`,
      "std_srvs/Trigger",
      {}
    );
  };

  const stopRunnerDetection = () => {
    ros.callService(
      `${nodeName}/stop_runner_detection`,
      "std_srvs/Trigger",
      {}
    );
  };

  const startRecordingVideo = () => {
    ros.callService(
      `${nodeName}/start_recording_video`,
      "std_srvs/Trigger",
      {}
    );
  };

  const stopRecordingVideo = () => {
    ros.callService(`${nodeName}/stop_recording_video`, "std_srvs/Trigger", {});
  };

  const saveImage = () => {
    ros.callService(`${nodeName}/save_image`, "std_srvs/Trigger", {});
  };

  return {
    connected: nodeState.connected,
    laserDetectionEnabled: nodeState.laser_detection_enabled,
    runnerDetectionEnabled: nodeState.runner_detection_enabled,
    recordingVideo: nodeState.recording_video,
    frameSrc,
    setExposure,
    startLaserDetection,
    stopLaserDetection,
    startRunnerDetection,
    stopRunnerDetection,
    startRecordingVideo,
    stopRecordingVideo,
    saveImage,
  };
}

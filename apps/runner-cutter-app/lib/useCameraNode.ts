import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

const DEVICE_STATES = ["disconnected", "connecting", "streaming"];
const INITIAL_STATE = {
  device_state: DEVICE_STATES[0],
  laser_detection_enabled: false,
  runner_detection_enabled: false,
  recording_video: false,
  interval_capture_active: false,
};

function convertToLocalReadableTime(secondsSinceEpoch: number) {
  const date = new Date(secondsSinceEpoch * 1000);
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

export default function useCameraNode(nodeName: string) {
  const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const [nodeState, setNodeState] = useState(INITIAL_STATE);
  const [logMessages, setLogMessages] = useState<string[]>([]);

  const nodeInfo: NodeInfo = useMemo(() => {
    return {
      name: nodeName,
      connected: rosbridgeNodeInfo.connected && nodeConnected,
      state: nodeState,
    };
  }, [nodeName, rosbridgeNodeInfo, nodeConnected, nodeState]);

  const getState = useCallback(async () => {
    const result = await ros.callService(
      `${nodeName}/get_state`,
      "camera_control_interfaces/GetState",
      {}
    );
    setNodeState({
      ...result.state,
      device_state: DEVICE_STATES[result.state.device_state.data],
    });
  }, [ros, nodeName, setNodeState]);

  const addLogMessage = useCallback(
    (logMessage: string) => {
      setLogMessages((prevLogMessages) => {
        const newLogMessages = [...prevLogMessages, logMessage];
        return newLogMessages.slice(-10);
      });
    },
    [setLogMessages]
  );

  // Initial node state
  useEffect(() => {
    const connected = ros.isNodeConnected(nodeName);
    if (connected) {
      getState();
    }
    setNodeConnected(connected);
  }, [ros, nodeName, getState, setNodeConnected]);

  // Subscriptions
  useEffect(() => {
    const onNodeConnectedSub = ros.onNodeConnected(
      (connectedNodeName, connected) => {
        if (connectedNodeName === nodeName) {
          setNodeConnected(connected);
          if (connected) {
            getState();
          } else {
            setNodeState(INITIAL_STATE);
          }
        }
      }
    );

    const stateSub = ros.subscribe(
      `${nodeName}/state`,
      "camera_control_interfaces/State",
      (message) => {
        setNodeState({
          ...message,
          device_state: DEVICE_STATES[message.device_state.data],
        });
      }
    );

    const logSub = ros.subscribe(
      `${nodeName}/log`,
      "rcl_interfaces/Log",
      (message) => {
        const timestamp_sec = parseInt(message["stamp"]["sec"]);
        const msg = `[${convertToLocalReadableTime(timestamp_sec)}] ${
          message["msg"]
        }`;
        addLogMessage(msg);
      }
    );

    return () => {
      onNodeConnectedSub.unsubscribe();
      stateSub.unsubscribe();
      logSub.unsubscribe();
    };
  }, [ros, nodeName, getState, setNodeConnected, setNodeState, addLogMessage]);

  const startDevice = useCallback(() => {
    ros.callService(`${nodeName}/start_device`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const closeDevice = useCallback(() => {
    ros.callService(`${nodeName}/close_device`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const setExposure = useCallback(
    (exposureUs: number) => {
      ros.callService(
        `${nodeName}/set_exposure`,
        "camera_control_interfaces/SetExposure",
        { exposure_us: exposureUs }
      );
    },
    [ros, nodeName]
  );

  const autoExposure = useCallback(() => {
    ros.callService(`${nodeName}/auto_exposure`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const startLaserDetection = useCallback(() => {
    ros.callService(
      `${nodeName}/start_laser_detection`,
      "std_srvs/Trigger",
      {}
    );
  }, [ros, nodeName]);

  const stopLaserDetection = useCallback(() => {
    ros.callService(`${nodeName}/stop_laser_detection`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const startRunnerDetection = useCallback(() => {
    ros.callService(
      `${nodeName}/start_runner_detection`,
      "std_srvs/Trigger",
      {}
    );
  }, [ros, nodeName]);

  const stopRunnerDetection = useCallback(() => {
    ros.callService(
      `${nodeName}/stop_runner_detection`,
      "std_srvs/Trigger",
      {}
    );
  }, [ros, nodeName]);

  const startRecordingVideo = useCallback(() => {
    ros.callService(
      `${nodeName}/start_recording_video`,
      "std_srvs/Trigger",
      {}
    );
  }, [ros, nodeName]);

  const stopRecordingVideo = useCallback(() => {
    ros.callService(`${nodeName}/stop_recording_video`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const startIntervalCapture = useCallback(
    (intervalSecs: number) => {
      ros.callService(
        `${nodeName}/start_interval_capture`,
        "camera_control_interfaces/StartIntervalCapture",
        { interval_secs: intervalSecs }
      );
    },
    [ros, nodeName]
  );

  const stopIntervalCapture = useCallback(() => {
    ros.callService(
      `${nodeName}/stop_interval_capture`,
      "std_srvs/Trigger",
      {}
    );
  }, [ros, nodeName]);

  const saveImage = useCallback(() => {
    ros.callService(`${nodeName}/save_image`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  return {
    nodeInfo,
    deviceState: nodeState.device_state,
    laserDetectionEnabled: nodeState.laser_detection_enabled,
    runnerDetectionEnabled: nodeState.runner_detection_enabled,
    recordingVideo: nodeState.recording_video,
    intervalCaptureActive: nodeState.interval_capture_active,
    logMessages,
    startDevice,
    closeDevice,
    setExposure,
    autoExposure,
    startLaserDetection,
    stopLaserDetection,
    startRunnerDetection,
    stopRunnerDetection,
    startRecordingVideo,
    stopRecordingVideo,
    startIntervalCapture,
    stopIntervalCapture,
    saveImage,
  };
}

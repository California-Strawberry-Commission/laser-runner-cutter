import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

const INITIAL_STATE = {
  connected: false,
  laser_detection_enabled: false,
  runner_detection_enabled: false,
  recording_video: false,
  interval_capture_active: false,
};

export default function useCameraNode(nodeName: string) {
  const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const [nodeState, setNodeState] = useState(INITIAL_STATE);

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
    setNodeState(result.state);
  }, [ros, nodeName, setNodeState]);

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
        setNodeState(message);
      }
    );

    const logSub = ros.subscribe(
      `${nodeName}/log`,
      "rcl_interfaces/Log",
      (message) => {
        console.log(`Received log: ${JSON.stringify(message)}`);
      }
    );

    return () => {
      onNodeConnectedSub.unsubscribe();
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, getState, setNodeConnected, setNodeState]);

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
    cameraConnected: nodeState.connected,
    laserDetectionEnabled: nodeState.laser_detection_enabled,
    runnerDetectionEnabled: nodeState.runner_detection_enabled,
    recordingVideo: nodeState.recording_video,
    intervalCaptureActive: nodeState.interval_capture_active,
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

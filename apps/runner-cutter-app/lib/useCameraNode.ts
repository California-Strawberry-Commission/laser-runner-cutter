import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

export default function useCameraNode(nodeName: string) {
  const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const [nodeState, setNodeState] = useState({
    connected: false,
    laser_detection_enabled: false,
    runner_detection_enabled: false,
    recording_video: false,
  });
  const [frameSrc, setFrameSrc] = useState<string>("");

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
    ros.onNodeConnected((connectedNodeName, connected) => {
      if (connectedNodeName === nodeName) {
        setNodeConnected(connected);
        if (connected) {
          getState();
        }
      }
    });

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
      // TODO: unsubscribe from ros.onNodeConnected
      stateSub.unsubscribe();
      frameSub.unsubscribe();
    };
  }, [ros, nodeName, getState, setNodeConnected, setNodeState, setFrameSrc]);

  const setExposure = useCallback(
    (exposureMs: number) => {
      ros.callService(
        `${nodeName}/set_exposure`,
        "camera_control_interfaces/SetExposure",
        { exposure_ms: exposureMs }
      );
    },
    [ros, nodeName]
  );

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

  const saveImage = useCallback(() => {
    ros.callService(`${nodeName}/save_image`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  return {
    nodeInfo,
    cameraConnected: nodeState.connected,
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

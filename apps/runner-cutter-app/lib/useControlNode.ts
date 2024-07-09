import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

const INITIAL_STATE = {
  calibrated: false,
  state: "idle",
};

export default function useControlNode(nodeName: string) {
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

  // Initial node connected state
  useEffect(() => {
    const connected = ros.isNodeConnected(nodeName);
    setNodeConnected(connected);
  }, [ros, nodeName, setNodeConnected]);

  // Subscriptions
  useEffect(() => {
    const onNodeConnectedSub = ros.onNodeConnected(
      (connectedNodeName, connected) => {
        if (connectedNodeName === nodeName) {
          setNodeConnected(connected);
        }
      }
    );

    const stateSub = ros.subscribe(
      `${nodeName}/state`,
      "runner_cutter_control_interfaces/State",
      (message) => {
        setNodeState(message);
      }
    );

    return () => {
      onNodeConnectedSub.unsubscribe();
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, setNodeConnected, setNodeState]);

  const calibrate = useCallback(() => {
    ros.callService(`${nodeName}/calibrate`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const addCalibrationPoint = useCallback(
    (normalizedX: number, normalizedY: number) => {
      ros.callService(
        `${nodeName}/add_calibration_points`,
        "runner_cutter_control_interfaces/AddCalibrationPoints",
        { normalized_pixel_coords: [{ x: normalizedX, y: normalizedY }] }
      );
    },
    [ros, nodeName]
  );

  const manualTargetAimLaser = useCallback(
    (normalizedX: number, normalizedY: number) => {
      ros.callService(
        `${nodeName}/manual_target_aim_laser`,
        "runner_cutter_control_interfaces/ManualTargetAimLaser",
        { normalized_pixel_coord: { x: normalizedX, y: normalizedY } }
      );
    },
    [ros, nodeName]
  );

  const startRunnerCutter = useCallback(() => {
    ros.callService(`${nodeName}/start_runner_cutter`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const stop = useCallback(() => {
    ros.callService(`${nodeName}/stop`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  return {
    nodeInfo,
    calibrated: nodeState.calibrated,
    controlState: nodeState.state,
    calibrate,
    addCalibrationPoint,
    manualTargetAimLaser,
    startRunnerCutter,
    stop,
  };
}

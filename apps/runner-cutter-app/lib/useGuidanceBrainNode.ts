import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

const INITIAL_STATE = {
  active: false,
};

export default function useGuidanceBrainNode(nodeName: string) {
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
      "guidance_brain_interfaces/GetState",
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
      "guidance_brain_interfaces/State",
      (message) => {
        setNodeState(message);
      }
    );

    return () => {
      onNodeConnectedSub.unsubscribe();
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, getState, setNodeConnected, setNodeState]);

  const setActive = useCallback((active: boolean) => {
    ros.callService(`${nodeName}/set_active`, "std_srvs/SetBool", { data: active });
  }, [ros, nodeName]);

  // const calibrate = useCallback(() => {
  //   ros.callService(`${nodeName}/calibrate`, "std_srvs/Trigger", {});
  // }, [ros, nodeName]);

  // const addCalibrationPoint = useCallback(
  //   (x: number, y: number) => {
  //     ros.callService(
  //       `${nodeName}/add_calibration_points`,
  //       "runner_cutter_control_interfaces/AddCalibrationPoints",
  //       { camera_pixels: [{ x, y }] }
  //     );
  //   },
  //   [ros, nodeName]
  // );

  // const manualTargetAimLaser = useCallback(
  //   (x: number, y: number) => {
  //     ros.callService(
  //       `${nodeName}/manual_target_aim_laser`,
  //       "runner_cutter_control_interfaces/ManualTargetAimLaser",
  //       { camera_pixel: { x, y } }
  //     );
  //   },
  //   [ros, nodeName]
  // );

  // const startRunnerCutter = useCallback(() => {
  //   ros.callService(`${nodeName}/start_runner_cutter`, "std_srvs/Trigger", {});
  // }, [ros, nodeName]);

  // const stop = useCallback(() => {
  //   ros.callService(`${nodeName}/stop`, "std_srvs/Trigger", {});
  // }, [ros, nodeName]);

  return {
    nodeInfo,
    connected: nodeConnected,
    nodeState,
    stop,
    setActive
  };
}
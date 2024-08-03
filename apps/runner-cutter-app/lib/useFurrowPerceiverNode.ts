import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

const INITIAL_STATE = {
  camera_connected: false,
  fps: 0,
};

export default function useFurrowPerceiverNode(nodeName: string) {
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
      "furrow_perceiver_interfaces/State",
      (message) => {
        setNodeState(message);
      }
    );

    return () => {
      onNodeConnectedSub.unsubscribe();
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, setNodeConnected, setNodeState]);

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
    // stop,
  };
}

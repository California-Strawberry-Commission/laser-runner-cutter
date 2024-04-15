import { useCallback, useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";

export default function useControlNode(nodeName: string) {
  const ros = useContext(ROSContext);
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const [nodeState, setNodeState] = useState({
    calibrated: false,
    state: "disconnected",
  });

  const getState = useCallback(async () => {
    const result = await ros.callService(
      `${nodeName}/get_state`,
      "runner_cutter_control_interfaces/GetState",
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
    ros.onNodeConnected(nodeName, (connected) => {
      setNodeConnected(connected);
      if (connected) {
        getState();
      }
    });

    const stateSub = ros.subscribe(
      `${nodeName}/state`,
      "runner_cutter_control_interfaces/State",
      (message) => {
        setNodeState(message);
      }
    );

    return () => {
      // TODO: unsubscribe from ros.onNodeConnected
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, getState, setNodeConnected, setNodeState]);

  const calibrate = () => {
    ros.callService(`${nodeName}/calibrate`, "std_srvs/Trigger", {});
  };

  const addCalibrationPoint = (x: number, y: number) => {
    ros.callService(
      `${nodeName}/add_calibration_points`,
      "runner_cutter_control_interfaces/AddCalibrationPoints",
      { camera_pixels: [{ x, y }] }
    );
  };

  const startRunnerCutter = () => {
    ros.callService(`${nodeName}/start_runner_cutter`, "std_srvs/Trigger", {});
  };

  const stop = () => {
    ros.callService(`${nodeName}/stop`, "std_srvs/Trigger", {});
  };

  return {
    nodeConnected,
    calibrated: nodeState.calibrated,
    controlState: nodeState.state,
    calibrate,
    addCalibrationPoint,
    startRunnerCutter,
    stop,
  };
}

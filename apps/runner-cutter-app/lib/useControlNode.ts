import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";

export default function useControlNode(nodeName: string) {
  const ros = useContext(ROSContext);
  const [nodeState, setNodeState] = useState({
    calibrated: false,
    state: "disconnected",
  });

  // Initial node state
  useEffect(() => {
    const getState = async () => {
      const result = await ros.callService(
        `${nodeName}/get_state`,
        "runner_cutter_control_interfaces/GetState",
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
      "runner_cutter_control_interfaces/State",
      (message) => {
        setNodeState(message);
      }
    );

    return () => {
      stateSub.unsubscribe();
    };
  }, [nodeName]);

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
    calibrated: nodeState.calibrated,
    controlState: nodeState.state,
    calibrate,
    addCalibrationPoint,
    startRunnerCutter,
    stop,
  };
}

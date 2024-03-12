import useROS from "@/lib/ros/useROS";

export default function useControlNode(nodeName: string) {
  const { callService } = useROS();

  const calibrate = () => {
    callService(`${nodeName}/calibrate`, "std_srvs/Empty", {});
  };

  const addCalibrationPoint = (x: number, y: number) => {
    callService(
      `${nodeName}/add_calibration_points`,
      "runner_cutter_control_interfaces/AddCalibrationPoints",
      { camera_pixels: [{ x, y }] }
    );
  };

  return { calibrate, addCalibrationPoint };
}

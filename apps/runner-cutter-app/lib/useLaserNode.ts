import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";

export const LASER_STATES = ["disconnected", "stopped", "playing"];

export default function useLaserNode(nodeName: string) {
  const ros = useContext(ROSContext);
  const [laserState, setLaserState] = useState<number>(0);

  // Initial node state
  useEffect(() => {
    const getState = async () => {
      const result = await ros.callService(
        `${nodeName}/get_state`,
        "laser_control_interfaces/GetState",
        {}
      );
      setLaserState(result.state.data);
    };
    getState();
  }, [nodeName]);

  // Subscriptions
  useEffect(() => {
    const stateSub = ros.subscribe(
      `${nodeName}/state`,
      "laser_control_interfaces/State",
      (message) => {
        setLaserState(message.data);
      }
    );
    return () => {
      stateSub.unsubscribe();
    };
  }, [nodeName]);

  const addPoint = (x: number, y: number) => {
    ros.callService(
      `${nodeName}/add_point`,
      "laser_control_interfaces/AddPoint",
      {
        point: {
          x: x,
          y: y,
        },
      }
    );
  };

  const clearPoints = () => {
    ros.callService(`${nodeName}/clear_points`, "std_srvs/Trigger", {});
  };

  const play = () => {
    ros.callService(`${nodeName}/play`, "std_srvs/Trigger", {});
  };

  const stop = () => {
    ros.callService(`${nodeName}/stop`, "std_srvs/Trigger", {});
  };

  const setColor = (r: number, g: number, b: number) => {
    ros.callService(
      `${nodeName}/set_color`,
      "laser_control_interfaces/SetColor",
      {
        r: r,
        g: g,
        b: b,
        i: 0.0,
      }
    );
  };

  return { laserState, addPoint, clearPoints, play, stop, setColor };
}

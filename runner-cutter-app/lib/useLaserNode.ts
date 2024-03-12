import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";

export const LASER_STATES = ["disconnected", "stopped", "playing"];

export default function useLaserNode(nodeName: string) {
  const { callService, subscribe } = useROS();
  const [laserState, setLaserState] = useState<number>(0);

  // Initial node state
  useEffect(() => {
    const getState = async () => {
      const result = await callService(
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
    const stateSub = subscribe(
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
    callService(`${nodeName}/add_point`, "laser_control_interfaces/AddPoint", {
      point: {
        x: x,
        y: y,
      },
    });
  };

  const clearPoints = () => {
    callService(`${nodeName}/clear_points`, "std_srvs/Empty", {});
  };

  const play = () => {
    callService(`${nodeName}/play`, "std_srvs/Empty", {});
  };

  const stop = () => {
    callService(`${nodeName}/stop`, "std_srvs/Empty", {});
  };

  const setColor = (r: number, g: number, b: number) => {
    callService(`${nodeName}/set_color`, "laser_control_interfaces/SetColor", {
      r: r,
      g: g,
      b: b,
      i: 0.0,
    });
  };

  return { laserState, addPoint, clearPoints, play, stop, setColor };
}

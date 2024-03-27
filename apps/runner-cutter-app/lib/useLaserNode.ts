import { useCallback, useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";

export const LASER_STATES = ["disconnected", "stopped", "playing"];

export default function useLaserNode(nodeName: string) {
  const ros = useContext(ROSContext);
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const [laserState, setLaserState] = useState<number>(0);

  const getState = useCallback(async () => {
    const result = await ros.callService(
      `${nodeName}/get_state`,
      "laser_control_interfaces/GetState",
      {}
    );
    setLaserState(result.state.data);
  }, [ros, nodeName, setLaserState]);

  // Initial node state
  useEffect(() => {
    const connected = ros.nodes.includes(nodeName);
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
      "laser_control_interfaces/State",
      (message) => {
        setLaserState(message.data);
      }
    );

    return () => {
      // TODO: unsubscribe from ros.onNodeConnected
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, getState, setNodeConnected, setLaserState]);

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

  return {
    nodeConnected,
    laserState,
    addPoint,
    clearPoints,
    play,
    stop,
    setColor,
  };
}
